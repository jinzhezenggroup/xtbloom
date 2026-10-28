"""Inspectable scheduling metadata, not a native cache or WARM identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BatchExecutionPlan:
    """Versioned, immutable plan for one finite host queue.

    Indices are canonical input ordinals, never object IDs or user labels.
    ``batches`` records the successful native calls after allocation retries;
    ``planned_batches`` retains the initial bounded schedule. Geometry and SCC
    options do not enter this scheduling hash: the plan is rebuilt each call,
    and native validation/cache/checkpoint identities remain authoritative.
    """

    requested_policy: str
    resolved_policy: str
    reason: str
    max_total_atoms: int | None
    atom_counts: tuple[int, ...]
    ao_counts: tuple[int, ...]
    basis_sha256: str | None
    grouping_plan_sha256: str | None
    planned_batches: tuple[tuple[int, ...], ...]
    batches: tuple[tuple[int, ...], ...]
    allocation_retries: tuple[tuple[int, ...], ...] = ()
    schema_version: int = 1

    @property
    def canonical_indices(self) -> tuple[int, ...]:
        """Execution position to original input ordinal, including every tail."""
        return tuple(index for batch in self.batches for index in batch)

    @property
    def inverse_indices(self) -> tuple[int, ...]:
        """Original input ordinal to its execution position."""
        inverse = [0] * len(self.atom_counts)
        for position, index in enumerate(self.canonical_indices):
            inverse[index] = position
        return tuple(inverse)

    @property
    def plan_sha256(self) -> str:
        """SHA256 of canonical UTF-8 JSON, including the actual retry schedule."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return detached serializable metadata without the self-referential hash."""
        document = asdict(self)
        document["canonical_indices"] = self.canonical_indices
        document["inverse_indices"] = self.inverse_indices
        return document


def atom_bounded_batches(
    atom_counts: tuple[int, ...], indices: tuple[int, ...], limit: int | None
) -> tuple[tuple[int, ...], ...]:
    """Slice indices without copying inputs; an oversized singleton is indivisible."""
    if limit is None:
        return (indices,)
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_atoms = 0
    for index in indices:
        atom_count = atom_counts[index]
        if current and current_atoms + atom_count > limit:
            batches.append(tuple(current))
            current = []
            current_atoms = 0
        current.append(index)
        current_atoms += atom_count
        if atom_count > limit:
            batches.append(tuple(current))
            current = []
            current_atoms = 0
    if current:
        batches.append(tuple(current))
    return tuple(batches)
