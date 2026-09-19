#!/usr/bin/env python3
"""Freeze the paper's dataset roles without copying or re-sampling raw data.

Corresponds to plan Sections 4 and 11 steps 1-2.  The script consumes the
already licensed canonical bundle manifests, verifies counts/disjointness and
content hashes, then writes a compact selection document.  It never edits an
input bundle and never drops malformed/unsupported rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


class FreezeError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_bundle_root(manifest: Path) -> Path:
    resolved = manifest.resolve()
    for parent in resolved.parents:
        if (parent / "SHA256SUMS").is_file():
            return parent
    raise FreezeError(f"manifest is not inside a hash-pinned bundle: {manifest}")


def verify_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FreezeError(f"bundle lacks SHA256SUMS: {root}")
    records = []
    listed_paths: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip(" *")
        if relative in listed_paths:
            raise FreezeError(f"bundle checksum lists a duplicate path: {relative}")
        listed_paths.add(relative)
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise FreezeError(f"bundle checksum escapes root: {relative}") from exc
        if not target.is_file():
            raise FreezeError(f"bundle checksum target is missing: {target}")
        actual = sha256(target)
        if actual != expected.lower():
            raise FreezeError(f"bundle checksum mismatch: {target}")
        records.append(
            {"path": relative, "bytes": target.stat().st_size, "sha256": actual}
        )
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if listed_paths != actual_paths:
        missing = sorted(actual_paths - listed_paths)
        extra = sorted(listed_paths - actual_paths)
        raise FreezeError(
            "bundle checksum inventory is not closed: "
            f"unlisted={missing[:16]} listed_but_absent={extra[:16]}"
        )
    licenses = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("LICENSE*")
        if path.is_file()
    )
    if not licenses:
        raise FreezeError(f"bundle has no retained dataset license: {root}")
    metadata = {}
    for relative in (
        "manifests/set_summary.json",
        "provenance/scan_summary.json",
        "provenance/sampling_config.json",
        "provenance/source_manifest.json",
    ):
        path = root / relative
        if path.is_file():
            metadata[relative] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "root": str(root),
        "sha256sums_sha256": sha256(checksum_path),
        "listed_files": records,
        "listed_file_count": len(records),
        "listed_bytes": sum(record["bytes"] for record in records),
        "closed_inventory": True,
        "license_files": licenses,
        "eligibility_and_sampling_metadata": metadata,
    }


def stable_key(seed: str, purpose: str, dataset: str, system_id: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{purpose}\0{dataset}\0{system_id}".encode()
    ).hexdigest()


def load(repo: Path, manifest: Path, dataset: str) -> list[Any]:
    from paper_runtime import install

    install(repo)
    import dataset_runner  # type: ignore

    return list(dataset_runner.load_manifest(manifest, dataset))


def features(item: Any) -> set[str]:
    assert item.system is not None
    row = item.system.manifest
    keys = (
        "domain",
        "charge_class",
        "spin_class",
        "element_class",
        "atom_count_bin",
        "gfn2_ao_bin",
        "geometry_class",
    )
    result = {f"subset={item.subset}"}
    for key in keys:
        value = row.get(key) or item.system.strata.get(key)
        if value:
            result.add(f"{key}={value}")
    return result


def diverse_selection(
    items: list[Any], count: int, seed: str, dataset: str
) -> list[Any]:
    candidates = [
        item for item in items if item.system is not None and item.subset == "main"
    ]
    if len(candidates) < count:
        raise FreezeError(f"{dataset} has only {len(candidates)} valid main rows")
    selected: list[Any] = []
    seen: Counter[str] = Counter()
    remaining = list(candidates)
    while len(selected) < count:
        remaining.sort(
            key=lambda item: (
                -sum(1 for value in features(item) if seen[value] == 0),
                stable_key(seed, "finite-difference", dataset, item.system_id),
            )
        )
        chosen = remaining.pop(0)
        selected.append(chosen)
        seen.update(features(chosen))
    return selected


def coordinate_selection(item: Any, count: int, seed: str) -> list[int]:
    assert item.system is not None
    extent = 3 * len(item.system.atomic_numbers)
    if extent < count:
        raise FreezeError(f"{item.system_id} has fewer than {count} coordinates")
    return sorted(
        range(extent),
        key=lambda index: hashlib.sha256(
            f"{seed}\0coordinate\0{item.system_id}\0{index}".encode()
        ).hexdigest(),
    )[:count]


def inventory(items: list[Any]) -> dict[str, Any]:
    subsets = Counter(item.subset for item in items)
    valid = Counter(item.subset for item in items if item.system is not None)
    malformed = Counter(item.subset for item in items if item.system is None)
    ids = [item.system_id for item in items]
    duplicates = sorted(system_id for system_id, n in Counter(ids).items() if n > 1)
    system_ids_by_subset: dict[str, list[str]] = {}
    for subset in sorted(subsets):
        system_ids_by_subset[subset] = sorted(
            item.system_id for item in items if item.subset == subset
        )
    return {
        "rows": len(items),
        "subsets": dict(sorted(subsets.items())),
        "valid": dict(sorted(valid.items())),
        "malformed": dict(sorted(malformed.items())),
        "duplicate_ids": duplicates,
        # The exact ID universe is part of the P0 completeness contract.  Counts
        # alone cannot detect a system omitted by every engine.
        "system_ids_by_subset": system_ids_by_subset,
    }


def validate_selected_contract(name: str, items: list[Any]) -> None:
    for item in items:
        if item.system is None:
            continue
        row = item.system.manifest
        probability_text = row.get("sampling_probability") or item.system.strata.get(
            "sampling_probability"
        )
        try:
            probability = float(probability_text)
        except (TypeError, ValueError) as exc:
            raise FreezeError(
                f"{name}/{item.system_id} lacks a numeric sampling_probability"
            ) from exc
        if not 0.0 < probability <= 1.0:
            raise FreezeError(
                f"{name}/{item.system_id} has invalid sampling_probability={probability}"
            )
        if not row.get("sampling_stratum"):
            raise FreezeError(f"{name}/{item.system_id} lacks sampling_stratum")
        if not row.get("gfn2_n_ao") or not row.get("gfn2_ao_bin"):
            raise FreezeError(f"{name}/{item.system_id} lacks frozen GFN2 AO metadata")
        digest_text = item.system.input_sha256
        if (
            digest_text is None
            or len(digest_text) != 64
            or any(
                character not in "0123456789abcdef" for character in digest_text.lower()
            )
        ):
            raise FreezeError(f"{name}/{item.system_id} lacks a valid input SHA-256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--qm9-main-manifest", type=Path, required=True)
    parser.add_argument("--qm9-performance-manifest", type=Path, required=True)
    parser.add_argument("--omol25-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    for path in (
        args.qm9_main_manifest,
        args.qm9_performance_manifest,
        args.omol25_manifest,
    ):
        if not path.is_file():
            raise FreezeError(f"manifest is missing: {path}")
    qm9_main = load(args.repo, args.qm9_main_manifest, "qm9")
    qm9_perf = load(args.repo, args.qm9_performance_manifest, "qm9")
    omol = load(args.repo, args.omol25_manifest, "omol25")
    inventories = {
        "qm9_main": inventory(qm9_main),
        "qm9_performance": inventory(qm9_perf),
        "omol25": inventory(omol),
    }
    for name, items in (
        ("qm9_main", qm9_main),
        ("qm9_performance", qm9_perf),
        ("omol25", omol),
    ):
        validate_selected_contract(name, items)
    for name, document in inventories.items():
        if document["duplicate_ids"]:
            raise FreezeError(f"{name} contains duplicate IDs")

    # The two QM9 files are independent frozen sets.  Require their labels to
    # match their paper roles so downstream runners cannot silently benchmark
    # the 10k equivalence set as the disjoint 2,048-system performance set.
    qm9_main_subsets = {item.subset for item in qm9_main}
    qm9_perf_subsets = {item.subset for item in qm9_perf}
    if qm9_main_subsets != {"main"}:
        raise FreezeError(
            f"QM9 main manifest has unexpected subset labels: {sorted(qm9_main_subsets)}"
        )
    if qm9_perf_subsets != {"performance"}:
        raise FreezeError(
            "QM9 performance manifest must label every selection as subset=performance; "
            f"found {sorted(qm9_perf_subsets)}"
        )

    qm9_main_ids = {item.system_id for item in qm9_main}
    qm9_perf_ids = {item.system_id for item in qm9_perf}
    if qm9_main_ids & qm9_perf_ids:
        raise FreezeError("QM9 main and performance selections overlap")
    omol_sets = {
        subset: {item.system_id for item in omol if item.subset == subset}
        for subset in ("main", "stress", "performance")
    }
    if any(
        omol_sets[a] & omol_sets[b]
        for a, b in (
            ("main", "stress"),
            ("main", "performance"),
            ("stress", "performance"),
        )
    ):
        raise FreezeError("OMol25 main/stress/performance selections overlap")

    expected = {
        "qm9_main": 10_000,
        "qm9_performance": 2_048,
        "omol25_main": 8_000,
        "omol25_stress": 2_000,
        "omol25_performance": 4_096,
    }
    actual = {
        "qm9_main": len(qm9_main),
        "qm9_performance": len(qm9_perf),
        "omol25_main": len(omol_sets["main"]),
        "omol25_stress": len(omol_sets["stress"]),
        "omol25_performance": len(omol_sets["performance"]),
    }
    if actual != expected:
        raise FreezeError(f"paper sample counts do not match plan: {actual}")

    fd = []
    for dataset, items in (("qm9", qm9_main), ("omol25", omol)):
        for item in diverse_selection(items, 16, args.seed, dataset):
            fd.append(
                {
                    "dataset": dataset,
                    "system_id": item.system_id,
                    "subset": item.subset,
                    "input_sha256": item.system.input_sha256,
                    "coordinate_indices": coordinate_selection(item, 8, args.seed),
                    "strata": item.system.strata,
                }
            )

    document = {
        "schema_version": 1,
        "purpose": "xTBloom paper plan frozen selections",
        "seed": args.seed,
        "expected_counts": expected,
        "actual_counts": actual,
        "manifests": {
            "qm9_main": {
                "path": str(args.qm9_main_manifest.resolve()),
                "sha256": sha256(args.qm9_main_manifest),
            },
            "qm9_performance": {
                "path": str(args.qm9_performance_manifest.resolve()),
                "sha256": sha256(args.qm9_performance_manifest),
            },
            "omol25": {
                "path": str(args.omol25_manifest.resolve()),
                "sha256": sha256(args.omol25_manifest),
            },
        },
        "bundles": {
            "qm9_main": verify_bundle(find_bundle_root(args.qm9_main_manifest)),
            "qm9_performance": verify_bundle(
                find_bundle_root(args.qm9_performance_manifest)
            ),
            "omol25": verify_bundle(find_bundle_root(args.omol25_manifest)),
        },
        "inventories": inventories,
        "finite_difference_selection": fd,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FreezeError(f"refusing to overwrite: {args.output}")
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
