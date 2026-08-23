#!/usr/bin/env python3
"""Build the current public cross-engine table from explicit evidence sources.

The publication manifest selects one clean evidence series per engine and
panel.  A CUDA optimization can therefore replace only ``xtbloom-cuda`` while
the unchanged CPU and third-party rows retain their original revisions.  The
generated CSV is a compact public index; issue-scoped bundles remain the
authoritative source for raw samples, correctness qualification, and commands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ENGINES = (
    "xtbloom-cpu",
    "xtbloom-cuda",
    "xtb",
    "tblite",
    "dxtb-cpu",
    "dxtb-cuda",
)
ENGINE_ORDER = {engine: index for index, engine in enumerate(SUPPORTED_ENGINES)}
HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

TABLE_COLUMNS = (
    "schema_version",
    "panel",
    "engine",
    "natoms",
    "batch_size",
    "start_policy",
    "effective_start_policy",
    "total_atoms_in_batch",
    "cpu_threads",
    "device_id",
    "availability",
    "median_ms",
    "mean_ms",
    "min_ms",
    "max_ms",
    "p95_ms",
    "systems_per_second_at_median",
    "correctness_status",
    "cross_engine_status",
    "max_abs_energy_delta_hartree",
    "max_abs_force_delta_hartree_per_bohr",
    "measured_date",
    "source_revision",
    "runtime_identity",
    "artifact_path",
    "artifact_sha256",
    "evidence_bundle",
    "reference_artifact_sha256",
    "hostname",
    "cpu_model",
    "cuda_device",
    "cuda_uuid",
    "cuda_driver",
    "protocol_id",
)


class PublicationError(RuntimeError):
    """An invalid or incomplete public benchmark selection."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_path(value: object, field: str) -> tuple[Path, str]:
    """Resolve one required repository-relative path without allowing escape."""
    resolved, relative = _repository_candidate_path(value, field)
    if not resolved.is_file():
        raise PublicationError(f"{field} does not exist: {value}")
    return resolved, relative


def _repository_candidate_path(value: object, field: str) -> tuple[Path, str]:
    """Resolve a safe repository path that may name an intentionally omitted file."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise PublicationError(f"{field} must be a repository-relative path")
    relative = Path(value)
    resolved = (REPOSITORY_ROOT / relative).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise PublicationError(f"{field} escapes the repository") from exc
    return resolved, relative.as_posix()


def _required_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicationError(f"{field} must be an object")
    return value


def _load_checksums(evidence_readme: Path) -> dict[str, str]:
    """Load the issue bundle's exact tracked-artifact checksum ledger."""
    checksums_path = evidence_readme.with_name("SHA256SUMS")
    try:
        lines = checksums_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublicationError(f"cannot read {checksums_path}: {exc}") from exc
    checksums: dict[str, str] = {}
    for line in lines:
        if not line:
            continue
        digest, separator, filename = line.partition("  ")
        if separator != "  " or HEX_64.fullmatch(digest) is None or not filename:
            raise PublicationError(f"invalid checksum record in {checksums_path}")
        if filename in checksums:
            raise PublicationError(f"duplicate checksum record: {filename}")
        checksums[filename] = digest
    return checksums


def _load_source_metadata(
    source: dict[str, Any],
    engine: str,
    evidence_readme: Path,
    checksums: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one checksummed evidence record instead of trusting the manifest."""
    metadata_path, _metadata_relative = _repository_path(
        source.get("metadata"), f"{engine}.metadata"
    )
    if metadata_path.parent != evidence_readme.parent:
        raise PublicationError(f"{engine} metadata is outside its evidence bundle")
    if checksums.get(metadata_path.name) != _sha256(metadata_path):
        raise PublicationError(f"{engine} metadata does not match SHA256SUMS")
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"cannot read {metadata_path}: {exc}") from exc
    if document.get("schema_version") != 1:
        raise PublicationError("publication metadata must use schema_version 1")
    raw_sources = document.get("sources")
    if not isinstance(raw_sources, list):
        raise PublicationError("publication metadata sources must be a list")
    matches = [
        item
        for item in raw_sources
        if isinstance(item, dict) and item.get("engine") == engine
    ]
    if len(matches) != 1:
        raise PublicationError(
            f"publication metadata must contain exactly one {engine} source"
        )
    return document, matches[0]


def _validate_checksummed_artifact(
    path: Path, expected_sha256: str | None = None
) -> None:
    """Require a retained evidence artifact to match its bundle checksum ledger."""
    checksums = _load_checksums(path.with_name("README.md"))
    actual_sha256 = _sha256(path)
    if checksums.get(path.name) != actual_sha256 or (
        expected_sha256 is not None and actual_sha256 != expected_sha256
    ):
        raise PublicationError(f"reference artifact is not checksum-qualified: {path}")


def _legacy_large_artifacts() -> dict[str, str]:
    """Load exact hashes for reproducible raw evidence omitted by the size policy."""
    ledger = REPOSITORY_ROOT / "benchmarks/evidence/legacy-large-artifacts.tsv"
    try:
        with ledger.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        raise PublicationError(f"cannot read {ledger}: {exc}") from exc
    bindings: dict[str, str] = {}
    for row in rows:
        path = row.get("path", "")
        digest = row.get("sha256", "")
        if not path or HEX_64.fullmatch(digest) is None or path in bindings:
            raise PublicationError(f"invalid record in {ledger}")
        bindings[path] = digest
    return bindings


def _validate_reference_binding(
    metadata_document: dict[str, Any],
    reference_sha256: str,
    engine: str,
    panel_id: str,
) -> None:
    """Bind a declared reference hash to retained or size-ledgered evidence."""
    bindings = _required_mapping(
        metadata_document.get("reference_bindings"), "reference_bindings"
    )
    binding = _required_mapping(
        bindings.get(reference_sha256), f"{engine}.{panel_id}.reference_binding"
    )
    kind = _required_string(binding, "kind")
    path, relative = _repository_candidate_path(
        binding.get("path"), f"{engine}.{panel_id}.reference_binding.path"
    )
    if kind == "retained-artifact":
        if path.is_file():
            _validate_checksummed_artifact(path, reference_sha256)
        elif _legacy_large_artifacts().get(relative) != reference_sha256:
            raise PublicationError(
                f"{engine}.{panel_id} reference is neither retained nor size-ledgered"
            )
        return
    if kind == "consumer-record":
        if not path.is_file():
            raise PublicationError(f"{engine}.{panel_id} consumer record is missing")
        _validate_checksummed_artifact(path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicationError(
                f"cannot read reference consumer record: {exc}"
            ) from exc
        recorded_sha256 = ((record.get("metadata") or {}).get("runner") or {}).get(
            "reference_json_sha256"
        )
        if recorded_sha256 != reference_sha256:
            raise PublicationError(
                f"{engine}.{panel_id} consumer record has a different reference"
            )
        return
    raise PublicationError(f"unsupported reference binding kind: {kind}")


def _panel_coordinates(
    metadata_document: dict[str, Any], panels: dict[str, tuple[int, str]]
) -> dict[str, set[int]]:
    """Load the complete coordinate set required from every selected artifact."""
    raw_coordinates = _required_mapping(
        metadata_document.get("panel_coordinates"), "panel_coordinates"
    )
    if set(raw_coordinates) != set(panels):
        raise PublicationError("publication metadata panel coordinates are incomplete")
    coordinates: dict[str, set[int]] = {}
    for panel_id, values in raw_coordinates.items():
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value <= 0 for value in values)
            or len(values) != len(set(values))
        ):
            raise PublicationError(f"invalid coordinates for panel {panel_id}")
        coordinates[panel_id] = set(values)
    return coordinates


def _required_string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise PublicationError(f"{field} must be a nonempty string")
    return value


def _positive_int(mapping: dict[str, Any], field: str) -> int:
    value = mapping.get(field)
    if type(value) is not int or value <= 0:
        raise PublicationError(f"{field} must be a positive integer")
    return value


def _finite_number(mapping: dict[str, Any], field: str) -> float:
    value = mapping.get(field)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise PublicationError(f"{field} must be finite")
    return float(value)


def _optional_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise PublicationError(f"invalid {field}: {value}") from exc
    if not math.isfinite(parsed):
        raise PublicationError(f"non-finite {field}: {value}")
    return parsed


def _optional_int(row: dict[str, str], field: str) -> int | None:
    value = row.get(field, "")
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise PublicationError(f"invalid {field}: {value}") from exc


def _validate_protocol(publication: dict[str, Any]) -> dict[str, Any]:
    protocol = _required_mapping(publication.get("protocol"), "publication.protocol")
    for field in ("warmups", "repetitions", "scc_max_iterations"):
        if field == "warmups":
            value = protocol.get(field)
            if type(value) is not int or value < 0:
                raise PublicationError("protocol.warmups must be a nonnegative integer")
        else:
            _positive_int(protocol, field)
    for field in (
        "cross_engine_energy_atol_hartree",
        "cross_engine_force_atol_hartree_per_bohr",
        "repeatability_energy_atol_hartree",
        "repeatability_force_atol_hartree_per_bohr",
        "perturb_sigma_bohr",
        "scc_charge_tolerance",
        "scc_energy_tolerance",
    ):
        value = _finite_number(protocol, field)
        if field == "perturb_sigma_bohr":
            if value <= 0.0:
                raise PublicationError("protocol.perturb_sigma_bohr must be positive")
        elif value < 0.0:
            raise PublicationError(f"protocol.{field} must be nonnegative")
    _required_mapping(protocol.get("convergence_contract"), "convergence_contract")
    return protocol


def load_publication(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate the selected public rows plus plot metadata."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"cannot read publication manifest: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise PublicationError("publication manifest must use schema_version 1")
    publication = _required_mapping(manifest.get("publication"), "publication")
    protocol_id = _required_string(publication, "protocol_id")
    protocol = _validate_protocol(publication)
    hardware = _required_mapping(publication.get("hardware"), "publication.hardware")
    hostname = _required_string(hardware, "hostname")
    cpu_model = _required_string(hardware, "cpu_model")
    cpu_threads = _positive_int(hardware, "cpu_threads")
    cuda_device = _required_mapping(hardware.get("cuda_device"), "hardware.cuda_device")
    cuda_name = _required_string(cuda_device, "name")
    cuda_uuid = _required_string(cuda_device, "uuid")
    cuda_driver = _required_string(cuda_device, "driver")
    expected_hardware = {
        "hostname": hostname,
        "cpu_model": cpu_model,
        "cpu_threads": cpu_threads,
        "cuda_device": {
            "name": cuda_name,
            "uuid": cuda_uuid,
            "driver": cuda_driver,
        },
    }

    raw_panels = publication.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise PublicationError("publication.panels must be a nonempty list")
    panels: dict[str, tuple[int, str]] = {}
    panel_order: dict[str, int] = {}
    for index, raw_panel in enumerate(raw_panels):
        panel = _required_mapping(raw_panel, f"publication.panels[{index}]")
        panel_id = _required_string(panel, "id")
        batch_size = _positive_int(panel, "batch_size")
        start_policy = _required_string(panel, "start_policy")
        if start_policy not in {"cold", "auto-warm"}:
            raise PublicationError(f"unsupported start policy: {start_policy}")
        if panel_id in panels:
            raise PublicationError(f"duplicate panel id: {panel_id}")
        panels[panel_id] = (batch_size, start_policy)
        panel_order[panel_id] = index

    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PublicationError("sources must be a nonempty list")
    rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    row_keys: set[tuple[str, int, int, str]] = set()
    source_engines: set[str] = set()
    for source_index, raw_source in enumerate(raw_sources):
        source = _required_mapping(raw_source, f"sources[{source_index}]")
        engine = _required_string(source, "engine")
        if engine not in SUPPORTED_ENGINES:
            raise PublicationError(f"unsupported engine: {engine}")
        if engine in source_engines:
            raise PublicationError(f"engine has multiple publication sources: {engine}")
        source_engines.add(engine)
        measured_date = _required_string(source, "measured_date")
        if ISO_DATE.fullmatch(measured_date) is None:
            raise PublicationError(f"invalid measured date for {engine}")
        source_revision = _required_string(source, "source_revision")
        if HEX_40.fullmatch(source_revision) is None:
            raise PublicationError(f"invalid source revision for {engine}")
        runtime_identity = _required_string(source, "runtime_identity")
        evidence_path, evidence_relative = _repository_path(
            source.get("evidence_bundle"), f"{engine}.evidence_bundle"
        )
        if evidence_path.name != "README.md":
            raise PublicationError(f"{engine} evidence bundle must name README.md")
        checksums = _load_checksums(evidence_path)
        metadata_document, source_metadata = _load_source_metadata(
            source, engine, evidence_path, checksums
        )
        if (
            metadata_document.get("protocol_id") != protocol_id
            or metadata_document.get("hardware") != expected_hardware
            or metadata_document.get("protocol") != protocol
        ):
            raise PublicationError(
                f"{engine} manifest does not match evidence publication metadata"
            )
        panel_coordinates = _panel_coordinates(metadata_document, panels)
        for field, expected in (
            ("measured_date", measured_date),
            ("source_revision", source_revision),
            ("runtime_identity", runtime_identity),
        ):
            if source_metadata.get(field) != expected:
                raise PublicationError(
                    f"{engine} manifest {field} does not match evidence metadata"
                )
        if (
            source_metadata.get("evidence_eligibility") != "eligible_clean_head"
            or source_metadata.get("source_dirty") is not False
            or source_metadata.get("runner_dirty") is not False
            or HEX_40.fullmatch(str(source_metadata.get("runner_revision", ""))) is None
        ):
            raise PublicationError(f"{engine} evidence metadata is not clean-eligible")
        raw_metadata_artifacts = source_metadata.get("artifacts")
        if not isinstance(raw_metadata_artifacts, list):
            raise PublicationError(f"{engine} evidence artifacts must be a list")
        metadata_artifacts: dict[str, dict[str, Any]] = {}
        for item in raw_metadata_artifacts:
            if not isinstance(item, dict):
                raise PublicationError(f"{engine} evidence artifact is not an object")
            item_panel = item.get("panel")
            if not isinstance(item_panel, str) or item_panel in metadata_artifacts:
                raise PublicationError(
                    f"{engine} evidence metadata has invalid panel records"
                )
            metadata_artifacts[item_panel] = item
        raw_artifacts = source.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise PublicationError(f"{engine} artifacts must be a nonempty list")
        seen_panels: set[str] = set()
        for artifact_index, raw_artifact in enumerate(raw_artifacts):
            artifact = _required_mapping(
                raw_artifact, f"{engine}.artifacts[{artifact_index}]"
            )
            panel_id = _required_string(artifact, "panel")
            if panel_id not in panels:
                raise PublicationError(f"{engine} uses unknown panel {panel_id}")
            if panel_id in seen_panels:
                raise PublicationError(f"{engine} duplicates panel {panel_id}")
            seen_panels.add(panel_id)
            batch_size, start_policy = panels[panel_id]
            metadata_artifact = metadata_artifacts.get(panel_id)
            if metadata_artifact is None:
                raise PublicationError(
                    f"{engine} evidence metadata omits panel {panel_id}"
                )
            csv_path, csv_relative = _repository_path(
                artifact.get("csv"), f"{engine}.{panel_id}.csv"
            )
            if csv_path.parent != evidence_path.parent:
                raise PublicationError(
                    f"{engine}.{panel_id} CSV is outside its evidence bundle"
                )
            if (
                metadata_artifact.get("csv") != csv_path.name
                or metadata_artifact.get("batch_size") != batch_size
                or metadata_artifact.get("start_policy") != start_policy
            ):
                raise PublicationError(
                    f"{engine}.{panel_id} does not match evidence metadata"
                )
            artifact_sha256 = _sha256(csv_path)
            if checksums.get(csv_path.name) != artifact_sha256:
                raise PublicationError(
                    f"{engine}.{panel_id} CSV does not match SHA256SUMS"
                )
            reference_sha256 = artifact.get("reference_artifact_sha256", "")
            metadata_reference_sha256 = metadata_artifact.get(
                "reference_artifact_sha256", ""
            )
            if metadata_reference_sha256 is None:
                metadata_reference_sha256 = ""
            if reference_sha256 != metadata_reference_sha256:
                raise PublicationError(
                    f"{engine}.{panel_id} reference does not match evidence metadata"
                )
            if engine == "tblite":
                if reference_sha256:
                    raise PublicationError(
                        "tblite reference rows cannot name a reference"
                    )
            elif (
                not isinstance(reference_sha256, str)
                or HEX_64.fullmatch(reference_sha256) is None
            ):
                raise PublicationError(
                    f"{engine}.{panel_id} requires a reference artifact SHA-256"
                )
            else:
                _validate_reference_binding(
                    metadata_document, reference_sha256, engine, panel_id
                )
            artifact_natoms: set[int] = set()
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for csv_row in reader:
                    if csv_row.get("engine") != engine:
                        raise PublicationError(
                            f"{csv_relative} contains engine {csv_row.get('engine')}"
                        )
                    if csv_row.get("job", ""):
                        raise PublicationError(
                            f"{csv_relative} contains non-panel jobs"
                        )
                    row_batch = _optional_int(csv_row, "batch_size")
                    natoms = _optional_int(csv_row, "natoms")
                    row_threads = _optional_int(csv_row, "cpu_threads")
                    if row_batch != batch_size or natoms is None or natoms <= 0:
                        raise PublicationError(
                            f"{csv_relative} has an invalid coordinate"
                        )
                    if row_threads != cpu_threads:
                        raise PublicationError(f"{csv_relative} changes the CPU budget")
                    artifact_natoms.add(natoms)
                    availability = csv_row.get("availability", "")
                    if availability not in {"available", "error", "unavailable"}:
                        raise PublicationError(
                            f"{csv_relative} has invalid availability {availability}"
                        )
                    median_ms = _optional_float(csv_row, "median_ms")
                    mean_ms = _optional_float(csv_row, "mean_ms")
                    min_ms = _optional_float(csv_row, "min_ms")
                    max_ms = _optional_float(csv_row, "max_ms")
                    p95_ms = _optional_float(csv_row, "p95_ms")
                    if availability == "available":
                        timing_values = (median_ms, mean_ms, min_ms, p95_ms)
                        if any(
                            value is None or value <= 0.0 for value in timing_values
                        ):
                            raise PublicationError(
                                f"{csv_relative} omits available timing statistics"
                            )
                        # Historical compact rows predate max_ms. Their fixed
                        # three-sample nearest-rank p95 is exactly the maximum.
                        if max_ms is None:
                            if protocol.get("repetitions") != 3:
                                raise PublicationError(
                                    f"{csv_relative} cannot reconstruct max_ms"
                                )
                            max_ms = p95_ms
                    key = (engine, natoms, batch_size, start_policy)
                    if key in row_keys:
                        raise PublicationError(f"duplicate publication row: {key}")
                    row_keys.add(key)
                    effective_start_policy = (
                        "cold" if engine.startswith("dxtb") else start_policy
                    )
                    cross_status = csv_row.get("cross_engine_status", "")
                    correctness_status = csv_row.get("correctness_status", "")
                    expected_cross_status = (
                        "reference" if engine == "tblite" else "pass"
                    )
                    if availability == "available" and (
                        correctness_status != "pass"
                        or cross_status != expected_cross_status
                    ):
                        raise PublicationError(
                            f"{csv_relative} contains unqualified available results"
                        )
                    internal_row = {
                        "engine": engine,
                        "natoms": natoms,
                        "batch_size": batch_size,
                        "job": None,
                        "start_policy": start_policy,
                        "effective_start_policy": effective_start_policy,
                        "_artifact_start_policy": start_policy,
                        "availability": availability,
                        "timing": {
                            "median_ms": median_ms,
                            "mean_ms": mean_ms,
                            "min_ms": min_ms,
                            "max_ms": max_ms,
                            "p95_ms": p95_ms,
                            "systems_per_second_at_median": _optional_float(
                                csv_row, "systems_per_second_at_median"
                            ),
                        },
                        "correctness": {
                            "status": correctness_status,
                            "cross_engine": {"status": cross_status},
                        },
                        "_publication_panel": panel_id,
                        "_source_revision": source_revision,
                        "_artifact_sha256": artifact_sha256,
                    }
                    rows.append(internal_row)
                    table_rows.append(
                        {
                            "schema_version": 1,
                            "panel": panel_id,
                            "engine": engine,
                            "natoms": natoms,
                            "batch_size": batch_size,
                            "start_policy": start_policy,
                            "effective_start_policy": effective_start_policy,
                            "total_atoms_in_batch": csv_row.get(
                                "total_atoms_in_batch", ""
                            ),
                            "cpu_threads": cpu_threads,
                            "device_id": csv_row.get("device_id", ""),
                            "availability": availability,
                            "median_ms": "" if median_ms is None else median_ms,
                            "mean_ms": "" if mean_ms is None else mean_ms,
                            "min_ms": "" if min_ms is None else min_ms,
                            "max_ms": "" if max_ms is None else max_ms,
                            "p95_ms": "" if p95_ms is None else p95_ms,
                            "systems_per_second_at_median": csv_row.get(
                                "systems_per_second_at_median", ""
                            ),
                            "correctness_status": correctness_status,
                            "cross_engine_status": cross_status,
                            "max_abs_energy_delta_hartree": csv_row.get(
                                "max_abs_energy_delta_hartree", ""
                            ),
                            "max_abs_force_delta_hartree_per_bohr": csv_row.get(
                                "max_abs_force_delta_hartree_per_bohr", ""
                            ),
                            "measured_date": measured_date,
                            "source_revision": source_revision,
                            "runtime_identity": runtime_identity,
                            "artifact_path": csv_relative,
                            "artifact_sha256": artifact_sha256,
                            "evidence_bundle": evidence_relative,
                            "reference_artifact_sha256": reference_sha256,
                            "hostname": hostname,
                            "cpu_model": cpu_model,
                            "cuda_device": cuda_name,
                            "cuda_uuid": cuda_uuid,
                            "cuda_driver": cuda_driver,
                            "protocol_id": protocol_id,
                        }
                    )
            if artifact_natoms != panel_coordinates[panel_id]:
                raise PublicationError(
                    f"{csv_relative} does not contain the declared coordinates"
                )
        missing_panels = set(panels) - seen_panels
        if missing_panels:
            raise PublicationError(
                f"{engine} omits publication panels: {sorted(missing_panels)}"
            )
        extra_metadata_panels = set(metadata_artifacts) - seen_panels
        if extra_metadata_panels:
            raise PublicationError(
                f"{engine} evidence metadata has extra panels: "
                f"{sorted(extra_metadata_panels)}"
            )
    missing_engines = set(SUPPORTED_ENGINES) - source_engines
    if missing_engines:
        raise PublicationError(f"publication omits engines: {sorted(missing_engines)}")

    sort_key = lambda row: (  # noqa: E731 - compact shared deterministic ordering
        panel_order[row["_publication_panel"]],
        ENGINE_ORDER[row["engine"]],
        row["natoms"],
    )
    rows.sort(key=sort_key)
    table_rows.sort(
        key=lambda row: (
            panel_order[str(row["panel"])],
            ENGINE_ORDER[str(row["engine"])],
            int(row["natoms"]),
        )
    )
    metadata = {
        "hardware": {
            "hostname": hostname,
            "cpu_model": cpu_model,
            "selected_cuda_device": {
                "runtime_uuid": cuda_uuid,
                "device": {
                    "uuid": cuda_uuid,
                    "name": cuda_name,
                    "driver": cuda_driver,
                },
            },
        },
        "threads": {"cpu_threads": cpu_threads},
        "protocol": protocol,
        "publication": {
            "protocol_id": protocol_id,
            "table_rows": table_rows,
        },
    }
    return rows, metadata


def render_table(metadata: dict[str, Any]) -> str:
    """Serialize the validated compact current-results table."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=TABLE_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows((metadata.get("publication") or {}).get("table_rows") or [])
    return buffer.getvalue()


def build_parser() -> argparse.ArgumentParser:
    """Build the current-results table command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the output differs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the publication manifest and write or check its table."""
    args = build_parser().parse_args(argv)
    try:
        _rows, metadata = load_publication(args.manifest)
        rendered = render_table(metadata)
        if args.check:
            current = args.output.read_text(encoding="utf-8")
            if current != rendered:
                raise PublicationError(f"generated table is stale: {args.output}")
            print(f"verified {args.output}")  # noqa: T201 - CLI completion output
            return 0
        args.output.write_text(rendered, encoding="utf-8", newline="")
        print(f"wrote {args.output}")  # noqa: T201 - CLI completion output
        return 0
    except (OSError, PublicationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
