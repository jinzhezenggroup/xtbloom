"""Deterministic finite-list GFN2 AO planning and result restoration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

try:
    from .convergence_grouping import RISK_BAND_ORDER
except ImportError:
    from convergence_grouping import RISK_BAND_ORDER

if TYPE_CHECKING:
    from pathlib import Path


class AOGroupingError(ValueError):
    """A finite-list AO plan or its result mapping is malformed."""


@dataclass(frozen=True)
class AOBatch:
    """One cap-bounded chunk whose systems retain their complete identity."""

    case_ids: tuple[str, ...]
    canonical_indices: tuple[int, ...]
    ao_count: int | None
    risk_band: str | None = None


@dataclass(frozen=True)
class AOGroupingPlan:
    """A deterministic ordering with an explicit inverse case-ID mapping."""

    strategy: str
    max_batch_size: int
    original_case_ids: tuple[str, ...]
    ordered_case_ids: tuple[str, ...]
    batches: tuple[AOBatch, ...]
    ao_counts_by_case_id: tuple[tuple[str, int], ...]
    basis_sha256: str | None
    plan_sha256: str
    risk_bands_by_case_id: tuple[tuple[str, str], ...] = ()
    risk_policy_sha256: str | None = None
    risk_freeze_plan_sha256: str | None = None

    @property
    def canonical_index_by_case_id(self) -> dict[str, int]:
        """Map each unique input identity back to its submitted position."""
        return {case_id: index for index, case_id in enumerate(self.original_case_ids)}


def load_gfn2_basis_ao_counts(path: Path) -> tuple[dict[int, int], str]:
    """Read per-element AO degeneracies from the generated GFN2 parameter JSON.

    Each parameter shell contributes ``2*l+1`` spatial AOs. Reading the
    generated shell angular momenta keeps this planner aligned with the actual
    model basis, including elements whose shell sets differ from their period.
    """
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise AOGroupingError(
            f"cannot read GFN2 basis parameters {path}: {exc}"
        ) from exc

    elements = document.get("elements")
    if not isinstance(elements, list) or not elements:
        raise AOGroupingError(f"GFN2 basis parameters {path} contain no elements")

    counts: dict[int, int] = {}
    for element in elements:
        if not isinstance(element, dict):
            raise AOGroupingError("GFN2 element parameter entry must be an object")
        atomic_number = element.get("atomic_number")
        shells = element.get("shells")
        if type(atomic_number) is not int or atomic_number <= 0:
            raise AOGroupingError("GFN2 element has an invalid atomic number")
        if atomic_number in counts:
            raise AOGroupingError(f"duplicate GFN2 element Z={atomic_number}")
        if not isinstance(shells, list) or not shells:
            raise AOGroupingError(f"GFN2 element Z={atomic_number} has no shells")
        ao_count = 0
        for shell in shells:
            angular_momentum = (
                shell.get("angular_momentum") if isinstance(shell, dict) else None
            )
            if type(angular_momentum) is not int or angular_momentum < 0:
                raise AOGroupingError(
                    f"GFN2 element Z={atomic_number} has invalid shell angular momentum"
                )
            ao_count += 2 * angular_momentum + 1
        counts[atomic_number] = ao_count

    return counts, hashlib.sha256(payload).hexdigest()


def count_gfn2_aos(
    atomic_numbers: tuple[int, ...] | list[int], basis_counts: dict[int, int]
) -> int:
    """Return the exact spatial AO count for one complete QM system."""
    if not atomic_numbers:
        raise AOGroupingError("a finite-list system must contain at least one atom")
    total = 0
    for atomic_number in atomic_numbers:
        if type(atomic_number) is not int:
            raise AOGroupingError("atomic numbers must be integers")
        try:
            total += basis_counts[atomic_number]
        except KeyError as exc:
            raise AOGroupingError(
                f"atomic number Z={atomic_number} is absent from GFN2 parameters"
            ) from exc
    return total


def _is_lowercase_sha256(value: object) -> bool:
    """Return whether a value is exactly 64 lowercase hexadecimal digits."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def make_plan(
    case_ids: tuple[str, ...] | list[str],
    max_batch_size: int,
    strategy: str = "original",
    ao_counts_by_case_id: dict[str, int] | None = None,
    basis_sha256: str | None = None,
    *,
    risk_bands_by_case_id: dict[str, str] | None = None,
    risk_policy_sha256: str | None = None,
    risk_freeze_plan_sha256: str | None = None,
) -> AOGroupingPlan:
    """Build a stable plan for a finite unique list without pruning or padding.

    Exact-AO ordering sorts by ascending AO count and then by canonical input
    index. The opt-in ``ao-risk`` strategy adds the frozen high-to-low risk
    band as a second bucket key and records both policy hashes. Every bucket
    keeps its own smaller tail; case identity and canonical scatter remain
    complete for all strategies.
    """
    ids = tuple(case_ids)
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise AOGroupingError("case IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise AOGroupingError("case IDs must be unique")
    if type(max_batch_size) is not int or max_batch_size <= 0:
        raise AOGroupingError("maximum batch size must be a positive integer")
    if strategy not in {"original", "exact-ao", "ao-risk"}:
        raise AOGroupingError(f"unknown AO grouping strategy: {strategy}")

    counts = ao_counts_by_case_id
    risk_bands = risk_bands_by_case_id
    if strategy != "ao-risk" and any(
        value is not None
        for value in (
            risk_bands_by_case_id,
            risk_policy_sha256,
            risk_freeze_plan_sha256,
        )
    ):
        raise AOGroupingError("risk metadata is only valid for ao-risk grouping")

    if strategy == "exact-ao":
        if counts is None:
            raise AOGroupingError("exact-AO grouping requires one AO count per case")
        if set(counts) != set(ids):
            raise AOGroupingError(
                "AO count IDs must match the finite case list exactly"
            )
        if any(type(count) is not int or count <= 0 for count in counts.values()):
            raise AOGroupingError("AO counts must be positive integers")
        if not isinstance(basis_sha256, str) or len(basis_sha256) != 64:
            raise AOGroupingError("exact-AO grouping requires the basis SHA-256")
    elif strategy == "ao-risk":
        if not isinstance(counts, dict) or set(counts) != set(ids):
            raise AOGroupingError("ao-risk grouping requires one AO count per case ID")
        if any(type(count) is not int or count <= 0 for count in counts.values()):
            raise AOGroupingError("AO counts must be positive integers")
        if not _is_lowercase_sha256(basis_sha256):
            raise AOGroupingError("ao-risk grouping requires a lowercase basis SHA-256")
        if not isinstance(risk_bands, dict) or set(risk_bands) != set(ids):
            raise AOGroupingError(
                "risk band IDs must match the finite case list exactly"
            )
        if any(
            not isinstance(band, str) or band not in RISK_BAND_ORDER
            for band in risk_bands.values()
        ):
            raise AOGroupingError("risk bands must be high, elevated, guarded, or low")
        if not _is_lowercase_sha256(risk_policy_sha256):
            raise AOGroupingError(
                "ao-risk grouping requires a lowercase risk policy SHA-256"
            )
        if not _is_lowercase_sha256(risk_freeze_plan_sha256):
            raise AOGroupingError(
                "ao-risk grouping requires a lowercase freeze plan SHA-256"
            )
    elif counts is not None and set(counts) != set(ids):
        raise AOGroupingError("AO count IDs must match the finite case list exactly")

    if strategy == "exact-ao":
        ordered_indices = tuple(
            sorted(
                range(len(ids)),
                key=lambda index: (counts[ids[index]], index),
            )
        )
    elif strategy == "ao-risk":
        risk_order = {band: index for index, band in enumerate(RISK_BAND_ORDER)}
        ordered_indices = tuple(
            sorted(
                range(len(ids)),
                key=lambda index: (
                    counts[ids[index]],
                    risk_order[risk_bands[ids[index]]],
                    index,
                ),
            )
        )
    else:
        ordered_indices = tuple(range(len(ids)))

    ordered_ids = tuple(ids[index] for index in ordered_indices)
    batches: list[AOBatch] = []
    if strategy == "original":
        for begin in range(0, len(ordered_indices), max_batch_size):
            chunk = ordered_indices[begin : begin + max_batch_size]
            batches.append(
                AOBatch(
                    case_ids=tuple(ids[index] for index in chunk),
                    canonical_indices=chunk,
                    ao_count=None,
                )
            )
    elif strategy == "exact-ao":
        group_begin = 0
        while group_begin < len(ordered_indices):
            ao_count = counts[ids[ordered_indices[group_begin]]]
            group_end = group_begin + 1
            while (
                group_end < len(ordered_indices)
                and counts[ids[ordered_indices[group_end]]] == ao_count
            ):
                group_end += 1
            for begin in range(group_begin, group_end, max_batch_size):
                chunk = ordered_indices[begin : min(begin + max_batch_size, group_end)]
                batches.append(
                    AOBatch(
                        case_ids=tuple(ids[index] for index in chunk),
                        canonical_indices=chunk,
                        ao_count=ao_count,
                    )
                )
            group_begin = group_end
    else:
        group_begin = 0
        while group_begin < len(ordered_indices):
            first_index = ordered_indices[group_begin]
            ao_count = counts[ids[first_index]]
            risk_band = risk_bands[ids[first_index]]
            group_end = group_begin + 1
            while group_end < len(ordered_indices):
                next_id = ids[ordered_indices[group_end]]
                if counts[next_id] != ao_count or risk_bands[next_id] != risk_band:
                    break
                group_end += 1
            # AO/risk buckets stay separate even when the current batch has room.
            for begin in range(group_begin, group_end, max_batch_size):
                chunk = ordered_indices[begin : min(begin + max_batch_size, group_end)]
                batches.append(
                    AOBatch(
                        case_ids=tuple(ids[index] for index in chunk),
                        canonical_indices=chunk,
                        ao_count=ao_count,
                        risk_band=risk_band,
                    )
                )
            group_begin = group_end

    count_pairs = (
        tuple((case_id, counts[case_id]) for case_id in ids)
        if counts is not None
        else ()
    )
    risk_pairs = (
        tuple((case_id, risk_bands[case_id]) for case_id in ids)
        if risk_bands is not None
        else ()
    )
    batch_documents = []
    for batch in batches:
        batch_document = {
            "case_ids": list(batch.case_ids),
            "canonical_indices": list(batch.canonical_indices),
            "ao_count": batch.ao_count,
        }
        if strategy == "ao-risk":
            batch_document["risk_band"] = batch.risk_band
        batch_documents.append(batch_document)
    hash_document = {
        "schema_version": 2 if strategy == "ao-risk" else 1,
        "strategy": strategy,
        "max_batch_size": max_batch_size,
        "original_case_ids": list(ids),
        "ao_counts_by_case_id": [list(pair) for pair in count_pairs],
        "basis_sha256": basis_sha256,
        "batches": batch_documents,
    }
    if strategy == "ao-risk":
        hash_document.update(
            {
                "risk_bands_by_case_id": [list(pair) for pair in risk_pairs],
                "risk_policy_sha256": risk_policy_sha256,
                "risk_freeze_plan_sha256": risk_freeze_plan_sha256,
            }
        )
    plan_sha256 = hashlib.sha256(
        json.dumps(hash_document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return AOGroupingPlan(
        strategy=strategy,
        max_batch_size=max_batch_size,
        original_case_ids=ids,
        ordered_case_ids=ordered_ids,
        batches=tuple(batches),
        ao_counts_by_case_id=count_pairs,
        basis_sha256=basis_sha256,
        plan_sha256=plan_sha256,
        risk_bands_by_case_id=risk_pairs,
        risk_policy_sha256=risk_policy_sha256,
        risk_freeze_plan_sha256=risk_freeze_plan_sha256,
    )


def split_batch_results(
    case_ids: tuple[str, ...],
    atom_offsets: list[int],
    point_charge_offsets: list[int],
    output: dict[str, Any],
    required_outputs: tuple[str, ...] = (),
) -> tuple[dict[str, Any], ...]:
    """Split packed public outputs into complete per-system records.

    Callers identify property arrays that must be present for their request;
    optional arrays remain optional for cases that do not request those values.
    """
    system_count = len(case_ids)
    per_system_keys = (
        "energies_hartree",
        "scc_iterations",
        "scc_converged",
        "per_system_status",
    )
    for key in per_system_keys:
        if key not in output or len(output[key]) != system_count:
            raise AOGroupingError(f"result field {key} does not match the batch size")
    for key in required_outputs:
        if key not in output:
            raise AOGroupingError(f"required result field {key} is missing")
    _validate_offsets(atom_offsets, system_count, "atom")
    _validate_offsets(point_charge_offsets, system_count, "point-charge")

    atom_count = atom_offsets[-1]
    point_count = point_charge_offsets[-1]
    _validate_packed_length(output, "atomic_charges_e", atom_count)
    _validate_packed_length(output, "forces_hartree_per_bohr", 3 * atom_count)
    _validate_packed_length(
        output, "point_charge_forces_hartree_per_bohr", 3 * point_count
    )

    records: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        atom_begin, atom_end = atom_offsets[index : index + 2]
        point_begin, point_end = point_charge_offsets[index : index + 2]
        record: dict[str, Any] = {
            "case_id": case_id,
            "energy_hartree": float(output["energies_hartree"][index]),
            "scc_iterations": int(output["scc_iterations"][index]),
            "scc_converged": int(output["scc_converged"][index]),
            "status": int(output["per_system_status"][index]),
        }
        if "forces_hartree_per_bohr" in output:
            begin, end = 3 * atom_begin, 3 * atom_end
            record["forces_hartree_per_bohr"] = [
                float(value) for value in output["forces_hartree_per_bohr"][begin:end]
            ]
        if "atomic_charges_e" in output:
            record["atomic_charges_e"] = [
                float(value)
                for value in output["atomic_charges_e"][atom_begin:atom_end]
            ]
        if "point_charge_forces_hartree_per_bohr" in output:
            begin, end = 3 * point_begin, 3 * point_end
            record["point_charge_forces_hartree_per_bohr"] = [
                float(value)
                for value in output["point_charge_forces_hartree_per_bohr"][begin:end]
            ]
        records.append(record)
    return tuple(records)


def scatter_case_results(
    plan: AOGroupingPlan,
    batch_results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Restore all output slices to canonical input order without filtering."""
    expected_ids = set(plan.original_case_ids)
    results_by_id: dict[str, dict[str, Any]] = {}
    for result in batch_results:
        case_id = result.get("case_id")
        if case_id not in expected_ids:
            raise AOGroupingError(f"result has unexpected case ID {case_id!r}")
        if case_id in results_by_id:
            raise AOGroupingError(f"duplicate result for case ID {case_id!r}")
        results_by_id[case_id] = result
    missing = [
        case_id for case_id in plan.original_case_ids if case_id not in results_by_id
    ]
    if missing:
        raise AOGroupingError("missing results for case IDs: " + ", ".join(missing))

    restored = []
    for index, case_id in enumerate(plan.original_case_ids):
        result = dict(results_by_id[case_id])
        result["original_index"] = index
        restored.append(result)
    return tuple(restored)


def _validate_offsets(offsets: list[int], system_count: int, label: str) -> None:
    if len(offsets) != system_count + 1 or not offsets or offsets[0] != 0:
        raise AOGroupingError(f"{label} offsets must contain batch_size + 1 entries")
    if any(type(value) is not int for value in offsets):
        raise AOGroupingError(f"{label} offsets must be integers")
    if any(left > right for left, right in pairwise(offsets)):
        raise AOGroupingError(f"{label} offsets must be monotonic")


def _validate_packed_length(output: dict[str, Any], key: str, expected: int) -> None:
    if key in output and len(output[key]) != expected:
        raise AOGroupingError(f"result field {key} has an invalid packed length")
