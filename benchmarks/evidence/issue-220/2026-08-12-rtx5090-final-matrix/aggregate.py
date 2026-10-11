#!/usr/bin/env python3
"""Aggregate issue #220 CUDA term evidence without dropping raw samples.

The term executables intentionally write one small artifact per N/B/topology
coordinate.  This evidence-local helper validates that the complete requested
grid is present and provenance-consistent, then combines those artifacts into
reviewable JSON/CSV files.  Input byte identities remain in the manifest so a
future maintainer can prove which raw files produced the retained bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from pathlib import Path
from typing import Any

ATOM_COUNTS = (16, 32, 48, 64, 96, 128, 256)
BATCH_SIZES = (1, 8, 32, 128)
TOPOLOGIES = ("compact", "open")
TERM_KINDS = ("d4", "aes2")
EXPECTED_TERMS = {
    "d4": (
        "d4_cn_cache_update",
        "d4_two_body_energy_potential",
        "d4_atm_energy",
        "d4_two_body_gradient",
        "d4_atm_gradient",
    ),
    "aes2": (
        "aes2_geometry",
        "aes2_potential",
        "aes2_energy",
        "aes2_vjp",
        "aes2_full",
    ),
}
PAIRLIST_MODES = ("sparse_build", "dense_build", "reuse")
CELL_PATTERN = re.compile(
    r"^(compact|open)-b(1|8|32|128)-n(16|32|48|64|96|128|256)\.json$"
)
HEX40_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class EvidenceError(RuntimeError):
    """Reject an incomplete or internally inconsistent evidence input."""


def sha256(path: Path) -> str:
    """Return the exact byte identity of one input or retained artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and retain the path in actionable failures."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"failed to load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected one JSON object in {path}")
    return value


def validate_provenance(document: dict[str, Any], context: str) -> None:
    """Require the auditable identity fields emitted by the final harness."""
    if document.get("identity_source") != "caller_supplied_and_archiver_verified":
        raise EvidenceError(f"{context} has an unverified identity source")
    if not HEX40_PATTERN.fullmatch(str(document.get("source_revision", ""))):
        raise EvidenceError(f"{context} has an invalid source revision")
    for field in ("executable_sha256", "build_identity_sha256"):
        if not HEX64_PATTERN.fullmatch(str(document.get(field, ""))):
            raise EvidenceError(f"{context} has an invalid {field}")
    if not isinstance(document.get("cuda_driver_version"), int):
        raise EvidenceError(f"{context} has no CUDA driver version")
    if not isinstance(document.get("device_name"), str) or not document["device_name"]:
        raise EvidenceError(f"{context} has no GPU model")
    if (
        not isinstance(document.get("compute_capability"), str)
        or not document["compute_capability"]
    ):
        raise EvidenceError(f"{context} has no compute capability")


def expected_coordinates() -> set[tuple[str, int, int]]:
    """Return the owner-approved isolated-term matrix coordinate set."""
    return {
        (topology, batch, atoms)
        for topology in TOPOLOGIES
        for atoms in ATOM_COUNTS
        for batch in BATCH_SIZES
    }


def linear_quantile(values: list[float], probability: float) -> float:
    """Return the deterministic type-7 linear sample quantile.

    The definition is explicit because the retained artifacts must remain
    reproducible without NumPy or an environment-dependent statistics stack.
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def distribution(values: object, context: str) -> dict[str, float | int]:
    """Validate one raw distribution and return compact robust statistics."""
    if not isinstance(values, list) or not values:
        raise EvidenceError(f"{context} has no raw samples")
    samples: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvidenceError(f"{context} contains a non-numeric sample")
        sample = float(value)
        if not math.isfinite(sample) or sample < 0.0:
            raise EvidenceError(f"{context} contains an invalid timing sample")
        samples.append(sample)
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    p25 = linear_quantile(samples, 0.25)
    p75 = linear_quantile(samples, 0.75)
    return {
        "sample_count": len(samples),
        "minimum_ms": min(samples),
        "p05_ms": linear_quantile(samples, 0.05),
        "median_ms": median,
        "mean_ms": mean,
        "p95_ms": linear_quantile(samples, 0.95),
        "maximum_ms": max(samples),
        "iqr_ms": p75 - p25,
        "mad_ms": statistics.median(abs(sample - median) for sample in samples),
        "cv": statistics.stdev(samples) / mean if len(samples) > 1 and mean else 0.0,
    }


def derived_term_row(kind: str, topology: str, row: dict[str, Any]) -> dict[str, Any]:
    """Build one compact D4/AES2 distribution row with explicit pair meaning."""
    stats = distribution(row.get("samples_ms"), f"{kind} {topology} {row.get('term')}")
    batch = row["batch"]
    return {
        "family": kind,
        "topology": topology,
        "batch": batch,
        "atoms_per_system": row["atoms_per_system"],
        "total_atoms": row["total_atoms"],
        "pair_count": row["pair_count"],
        "pair_count_semantics": row["pair_count_semantics"],
        "dense_pair_count": row["dense_pair_count"],
        "term_or_mode": row["term"],
        "workload": row["workload"],
        **stats,
        "median_us_per_system": 1000.0 * stats["median_ms"] / batch,
        "validation_status": "pass",
        "max_abs_error": None,
    }


def aggregate_term_kind(
    input_root: Path, output_dir: Path, kind: str
) -> dict[str, Any]:
    """Validate and combine one complete D4 or AES2 term matrix."""
    source_dir = input_root / "terms" / kind
    paths = sorted(source_dir.glob("*.json"))
    if len(paths) != len(expected_coordinates()):
        expected_cells = len(expected_coordinates())
        raise EvidenceError(
            f"{kind} matrix has {len(paths)} JSON cells; expected {expected_cells}"
        )

    cells: list[dict[str, Any]] = []
    coordinates: set[tuple[str, int, int]] = set()
    provenance: dict[str, Any] | None = None
    input_artifacts: list[dict[str, Any]] = []
    for path in paths:
        match = CELL_PATTERN.fullmatch(path.name)
        if match is None:
            raise EvidenceError(f"unexpected {kind} cell filename: {path.name}")
        topology, batch_text, atoms_text = match.groups()
        batch = int(batch_text)
        atoms = int(atoms_text)
        coordinate = (topology, batch, atoms)
        if coordinate in coordinates:
            raise EvidenceError(f"duplicate {kind} coordinate: {coordinate}")
        coordinates.add(coordinate)

        document = load_json(path)
        validate_provenance(document, str(path))
        cell_provenance = {
            name: document.get(name)
            for name in (
                "schema_version",
                "benchmark",
                "timing_scope",
                "warmups",
                "samples_per_term",
                "profile_range_scope",
                "source_revision",
                "executable_sha256",
                "build_identity_sha256",
                "identity_source",
                "cuda_header_version",
                "cuda_runtime_version",
                "cuda_driver_version",
                "device_id",
                "device_name",
                "compute_capability",
            )
        }
        if provenance is None:
            provenance = cell_provenance
        elif provenance != cell_provenance:
            raise EvidenceError(f"inconsistent {kind} provenance in {path}")
        if document.get("topology") != topology:
            raise EvidenceError(f"{kind} topology does not match filename: {path}")
        if document.get("warmups", 0) < 3 or document.get("samples_per_term", 0) < 20:
            raise EvidenceError(
                f"{kind} protocol is below 3 warmups/20 samples: {path}"
            )

        rows = document.get("rows")
        if not isinstance(rows, list) or len(rows) != 5:
            raise EvidenceError(
                f"{kind} cell must contain exactly five term rows: {path}"
            )
        if (
            tuple(row.get("term") for row in rows if isinstance(row, dict))
            != EXPECTED_TERMS[kind]
        ):
            raise EvidenceError(
                f"{kind} cell has an unexpected term set or order: {path}"
            )
        for row in rows:
            if not isinstance(row, dict):
                raise EvidenceError(f"{kind} term row is not an object: {path}")
            if row.get("batch") != batch or row.get("atoms_per_system") != atoms:
                raise EvidenceError(
                    f"{kind} row coordinate does not match filename: {path}"
                )
            expected_semantics = (
                "committed_50_bohr_retained_pairs"
                if kind == "d4"
                else "packed_all_pairs"
            )
            if row.get("pair_count_semantics") != expected_semantics:
                raise EvidenceError(
                    f"{kind} row has incorrect pair-count semantics: {path}"
                )
            pair_count = row.get("pair_count")
            dense_pair_count = row.get("dense_pair_count")
            if (
                not isinstance(pair_count, int)
                or not isinstance(dense_pair_count, int)
                or pair_count < 0
                or dense_pair_count < pair_count
                or (kind == "aes2" and pair_count != dense_pair_count)
            ):
                raise EvidenceError(f"{kind} row has invalid pair counts: {path}")
            samples = row.get("samples_ms")
            if (
                not isinstance(samples, list)
                or len(samples) != document["samples_per_term"]
            ):
                raise EvidenceError(
                    f"{kind} row has an incomplete sample distribution: {path}"
                )
            stats = distribution(samples, f"{kind} row in {path}")
            if (
                row.get("minimum_ms") != stats["minimum_ms"]
                or row.get("maximum_ms") != stats["maximum_ms"]
            ):
                raise EvidenceError(
                    f"{kind} row min/max does not match raw samples: {path}"
                )
            # The measured source revision used the upper middle sample for
            # even distributions. Raw samples are authoritative, so normalize
            # the retained median to the repository's conventional definition.
            row["median_ms"] = stats["median_ms"]

        cells.append(
            {
                "topology": topology,
                "batch": batch,
                "atoms_per_system": atoms,
                "post_timing_validation": "pass",
                "argv": document.get("argv"),
                "rows": rows,
            }
        )
        input_artifacts.append(
            {
                "path": str(path.relative_to(input_root.parent.parent)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        csv_path = path.with_suffix(".csv")
        if not csv_path.is_file():
            raise EvidenceError(f"missing paired CSV artifact: {csv_path}")
        input_artifacts.append(
            {
                "path": str(csv_path.relative_to(input_root.parent.parent)),
                "bytes": csv_path.stat().st_size,
                "sha256": sha256(csv_path),
            }
        )

    missing = expected_coordinates() - coordinates
    extra = coordinates - expected_coordinates()
    if missing or extra:
        raise EvidenceError(
            f"{kind} coordinate mismatch: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    assert provenance is not None
    cells.sort(
        key=lambda cell: (cell["topology"], cell["atoms_per_system"], cell["batch"])
    )
    combined = {
        "schema_version": 1,
        "benchmark": f"xtbloom_cuda_{kind}_term_matrix",
        "derivation": (
            "Lossless row/sample aggregation of successful per-cell benchmark "
            "artifacts; each source executable writes results only after "
            "post-timing validation passes. Medians are recomputed from raw "
            "samples using the conventional mean of two middle values."
        ),
        "matrix": {
            "topologies": list(TOPOLOGIES),
            "atom_counts": list(ATOM_COUNTS),
            "batch_sizes": list(BATCH_SIZES),
            "coordinates": len(cells),
            "term_rows": sum(len(cell["rows"]) for cell in cells),
            "raw_samples": sum(
                len(row["samples_ms"]) for cell in cells for row in cell["rows"]
            ),
        },
        "provenance": provenance,
        "cells": cells,
    }
    json_path = output_dir / f"{kind}-term-matrix.json"
    json_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / f"{kind}-term-matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=(
                "topology",
                "batch",
                "atoms_per_system",
                "term",
                "workload",
                "total_atoms",
                "pair_count",
                "pair_count_semantics",
                "dense_pair_count",
                "minimum_ms",
                "median_ms",
                "maximum_ms",
                "samples",
                "post_timing_validation",
            ),
        )
        writer.writeheader()
        for cell in cells:
            for row in cell["rows"]:
                writer.writerow(
                    {
                        "topology": cell["topology"],
                        "batch": cell["batch"],
                        "atoms_per_system": cell["atoms_per_system"],
                        "term": row["term"],
                        "workload": row["workload"],
                        "total_atoms": row["total_atoms"],
                        "pair_count": row["pair_count"],
                        "pair_count_semantics": row["pair_count_semantics"],
                        "dense_pair_count": row["dense_pair_count"],
                        "minimum_ms": row["minimum_ms"],
                        "median_ms": row["median_ms"],
                        "maximum_ms": row["maximum_ms"],
                        "samples": len(row["samples_ms"]),
                        "post_timing_validation": cell["post_timing_validation"],
                    }
                )
    return {
        "kind": kind,
        "inputs": input_artifacts,
        "outputs": [json_path.name, csv_path.name],
    }


def copy_pairlist(input_root: Path, output_dir: Path) -> dict[str, Any]:
    """Retain the authoritative pair-list JSON and its paired compact CSV."""
    inputs = []
    outputs = []
    for name in ("pairlist-matrix.json", "pairlist-matrix.csv"):
        source = input_root / name
        if not source.is_file():
            raise EvidenceError(f"missing pair-list artifact: {source}")
        destination = output_dir / name
        shutil.copyfile(source, destination)
        inputs.append(
            {
                "path": str(source.relative_to(input_root.parent.parent)),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
        outputs.append(destination.name)
    document = load_json(output_dir / "pairlist-matrix.json")
    validate_provenance(document, "pair-list JSON")
    protocol = document.get("protocol", {})
    if protocol.get("warmups", 0) < 3 or protocol.get("samples_per_cell", 0) < 20:
        raise EvidenceError("pair-list protocol is below 3 warmups/20 samples")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_coordinates()):
        raise EvidenceError(
            "pair-list JSON does not contain the complete 56-cell matrix"
        )
    coordinates = {
        (row.get("topology"), row.get("batch"), row.get("atoms")) for row in rows
    }
    if coordinates != {
        ("sparse" if t == "open" else t, b, n) for t, b, n in expected_coordinates()
    }:
        raise EvidenceError("pair-list JSON coordinate set is incomplete or duplicated")
    if any(row.get("validation", {}).get("status") != "pass" for row in rows):
        raise EvidenceError("pair-list matrix contains a non-passing validation row")
    for row in rows:
        for mode in PAIRLIST_MODES:
            stats = distribution(
                row.get(mode, {}).get("samples_ms"), f"pair-list {mode}"
            )
            if stats["sample_count"] != protocol["samples_per_cell"]:
                raise EvidenceError(
                    "pair-list matrix contains an incomplete raw distribution"
                )
    return {"kind": "pairlist", "inputs": inputs, "outputs": outputs}


def pairlist_distribution_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one robust-statistics row for every pair-list mode and cell."""
    rows: list[dict[str, Any]] = []
    for source in document["rows"]:
        topology = "open" if source["topology"] == "sparse" else source["topology"]
        batch = source["batch"]
        atoms = source["atoms"]
        retained = source["validation"]["retained_pairs"]
        dense = source["validation"]["dense_pairs"]
        sparse_median = distribution(
            source["sparse_build"]["samples_ms"], "pair-list sparse build"
        )["median_ms"]
        dense_median = distribution(
            source["dense_build"]["samples_ms"], "pair-list dense build"
        )["median_ms"]
        selected_median = dense_median if atoms <= 40 else sparse_median
        best_median = min(sparse_median, dense_median)
        reuse_median = distribution(source["reuse"]["samples_ms"], "pair-list reuse")[
            "median_ms"
        ]
        for mode in PAIRLIST_MODES:
            stats = distribution(source[mode]["samples_ms"], f"pair-list {mode}")
            active_pairs = dense if mode == "dense_build" else retained
            rows.append(
                {
                    "family": "pairlist",
                    "topology": topology,
                    "batch": batch,
                    "atoms_per_system": atoms,
                    "total_atoms": atoms * batch,
                    "dense_pair_extent": dense,
                    "active_pairs": active_pairs,
                    "term_or_mode": mode,
                    "workload": "50_bohr_pairlist",
                    **stats,
                    "median_us_per_system": 1000.0 * stats["median_ms"] / batch,
                    "median_ns_per_active_pair": (
                        1_000_000.0 * stats["median_ms"] / active_pairs
                        if active_pairs
                        else None
                    ),
                    "retained_pair_fraction": retained / dense if dense else 0.0,
                    "sparse_vs_dense_delta_percent": 100.0
                    * (sparse_median / dense_median - 1.0),
                    "reuse_vs_selected_build_delta_percent": 100.0
                    * (reuse_median / selected_median - 1.0),
                    "selected_by_40_atom_policy": (
                        mode == ("dense_build" if atoms <= 40 else "sparse_build")
                    ),
                    "policy_regret_percent": 100.0
                    * (selected_median / best_median - 1.0),
                    "validation_status": source["validation"]["status"],
                    "max_abs_error": source["validation"]["max_abs_coordination_error"],
                }
            )
    return rows


def paired_topology_ratios(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    """Summarize compact/open ratios for every term in one family."""
    selected = [row for row in rows if row["family"] == family]
    by_key = {
        (
            row["topology"],
            row["batch"],
            row["atoms_per_system"],
            row["term_or_mode"],
        ): row
        for row in selected
    }
    summary: dict[str, Any] = {}
    for term in EXPECTED_TERMS[family]:
        ratios = []
        for batch in BATCH_SIZES:
            for atoms in ATOM_COUNTS:
                compact = by_key[("compact", batch, atoms, term)]["median_ms"]
                open_value = by_key[("open", batch, atoms, term)]["median_ms"]
                ratios.append(
                    {
                        "batch": batch,
                        "atoms_per_system": atoms,
                        "compact_over_open": compact / open_value,
                    }
                )
        ratio_values = [row["compact_over_open"] for row in ratios]
        maximum = max(ratios, key=lambda row: row["compact_over_open"])
        summary[term] = {
            "minimum_compact_over_open": min(ratio_values),
            "median_compact_over_open": statistics.median(ratio_values),
            "maximum_compact_over_open": maximum["compact_over_open"],
            "maximum_coordinate": {
                "batch": maximum["batch"],
                "atoms_per_system": maximum["atoms_per_system"],
            },
        }
    return summary


def write_distribution_summary(
    output_dir: Path, term_documents: dict[str, dict[str, Any]]
) -> list[str]:
    """Write compact exact-distribution rows plus dispatch/topology decisions."""
    pairlist = load_json(output_dir / "pairlist-matrix.json")
    rows = pairlist_distribution_rows(pairlist)
    for kind, document in term_documents.items():
        for cell in document["cells"]:
            rows.extend(
                derived_term_row(kind, cell["topology"], row) for row in cell["rows"]
            )
    rows.sort(
        key=lambda row: (
            row["family"],
            row["topology"],
            row["atoms_per_system"],
            row["batch"],
            row["term_or_mode"],
        )
    )

    boundary = []
    pair_rows = [row for row in rows if row["family"] == "pairlist"]
    pair_index = {
        (
            row["topology"],
            row["batch"],
            row["atoms_per_system"],
            row["term_or_mode"],
        ): row
        for row in pair_rows
    }
    for topology in TOPOLOGIES:
        for atoms in (32, 48):
            for batch in BATCH_SIZES:
                sparse = pair_index[(topology, batch, atoms, "sparse_build")]
                dense = pair_index[(topology, batch, atoms, "dense_build")]
                boundary.append(
                    {
                        "topology": topology,
                        "atoms_per_system": atoms,
                        "batch": batch,
                        "sparse_median_ms": sparse["median_ms"],
                        "dense_median_ms": dense["median_ms"],
                        "sparse_vs_dense_delta_percent": sparse[
                            "sparse_vs_dense_delta_percent"
                        ],
                    }
                )
    nonoptimal = [
        row
        for row in pair_rows
        if row["term_or_mode"] == "sparse_build" and row["policy_regret_percent"] > 0.0
    ]
    max_regret = max(nonoptimal, key=lambda row: row["policy_regret_percent"])
    reuse_spikes = sorted(
        (row for row in pair_rows if row["term_or_mode"] == "reuse"),
        key=lambda row: row["maximum_ms"] / row["median_ms"],
        reverse=True,
    )
    decisions = {
        "distribution_definition": {
            "median": "arithmetic mean of the two central ordered samples for even n",
            "p05_p95": "type-7 linear quantiles at probabilities 0.05 and 0.95",
            "iqr": "type-7 p75 minus p25",
            "mad": "median absolute deviation from the derived median",
            "cv": "sample standard deviation divided by arithmetic mean",
        },
        "completeness": {
            "pairlist_cells": 56,
            "pairlist_distribution_rows": len(pair_rows),
            "pairlist_raw_samples": sum(row["sample_count"] for row in pair_rows),
            "d4_term_rows": sum(row["family"] == "d4" for row in rows),
            "d4_raw_samples": sum(
                row["sample_count"] for row in rows if row["family"] == "d4"
            ),
            "aes2_term_rows": sum(row["family"] == "aes2" for row in rows),
            "aes2_raw_samples": sum(
                row["sample_count"] for row in rows if row["family"] == "aes2"
            ),
        },
        "dispatch_40_atoms": {
            "decision": "retain",
            "reason": (
                "Open topology already favors sparse at N=32, while compact "
                "topology still favors dense at N=48 for every measured batch. "
                "A topology-agnostic atom-only threshold cannot improve both "
                "distributions, and no N=40 neighborhood was run."
            ),
            "boundary_rows": boundary,
            "nonoptimal_cells_under_current_policy": len(nonoptimal),
            "maximum_policy_regret_percent": max_regret["policy_regret_percent"],
            "maximum_regret_coordinate": {
                "topology": max_regret["topology"],
                "batch": max_regret["batch"],
                "atoms_per_system": max_regret["atoms_per_system"],
            },
        },
        "reuse_distribution": {
            "warning": (
                "Several open/sparse reuse cells are multimodal or contain "
                "periodic spikes; min/median/max alone is not a sufficient archive."
            ),
            "largest_max_over_median": [
                {
                    "topology": row["topology"],
                    "batch": row["batch"],
                    "atoms_per_system": row["atoms_per_system"],
                    "median_ms": row["median_ms"],
                    "p05_ms": row["p05_ms"],
                    "p95_ms": row["p95_ms"],
                    "maximum_ms": row["maximum_ms"],
                    "max_over_median": row["maximum_ms"] / row["median_ms"],
                    "cv": row["cv"],
                }
                for row in reuse_spikes[:12]
            ],
        },
        "compact_over_open": {
            "d4": paired_topology_ratios(rows, "d4"),
            "aes2": paired_topology_ratios(rows, "aes2"),
        },
        "limitations": [
            (
                "No N=40 or dense neighborhood at N=36/44 was measured, so the "
                "exact crossover cannot be refit."
            ),
            (
                "The isolated matrices are homogeneous open/compact fixtures; "
                "heterogeneous term timing was not run."
            ),
            (
                "D4 rows record committed 50-bohr retained pairs and dense pair "
                "extents, but not per-role active CN/two-body pairs or ATM-triple "
                "counts; no ns/active-interaction normalization is claimed."
            ),
            (
                "Term rows are prepared internal production entry points, not "
                "complete public SCC/inference calls."
            ),
            (
                "Nsight Compute DRAM/occupancy counters are unavailable with "
                "ERR_NVGPUCTRPERM."
            ),
            (
                "The cancelled long public N/B/topology/QM-MM Cartesian sweep "
                "remains not run and outside revised closure scope."
            ),
        ],
    }
    json_path = output_dir / "distribution-summary.json"
    json_path.write_text(
        json.dumps(
            {"schema_version": 1, "decisions": decisions, "rows": rows}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "distribution-summary.csv"
    fieldnames = tuple(
        dict.fromkeys(key for row in rows for key in row if key not in {"samples_ms"})
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return [json_path.name, csv_path.name]


def parse_args() -> argparse.Namespace:
    """Parse explicit build input and retained output roots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Generate the compact matrix and a byte-identity manifest."""
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [copy_pairlist(input_root, output_dir)]
    artifacts.extend(
        aggregate_term_kind(input_root, output_dir, kind) for kind in TERM_KINDS
    )
    term_documents = {
        kind: load_json(output_dir / f"{kind}-term-matrix.json") for kind in TERM_KINDS
    }
    summary_outputs = write_distribution_summary(output_dir, term_documents)
    manifest = {
        "schema_version": 1,
        "generator": "aggregate.py",
        "input_root": "build/issue220-final-measurements",
        "matrix": {
            "topologies": list(TOPOLOGIES),
            "atom_counts": list(ATOM_COUNTS),
            "batch_sizes": list(BATCH_SIZES),
        },
        "artifacts": artifacts,
        "derived_outputs": summary_outputs,
    }
    (output_dir / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
