"""Freeze an input-only, interpretable AO and convergence-risk policy.

The prototype deliberately produces per-case annotations rather than a batch
permutation. The finite-list AO planner owns ordering, caps, and inverse result
mapping so the full input is planned exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ConvergenceGroupingError(ValueError):
    """A frozen convergence-grouping input or calibration guard is invalid."""


SCHEMA_VERSION = 1
SPLIT_SEED = "xtbloom-issue-514-fixed-group-split-v1-2026-10-09"
HOLDOUT_FRACTION_NUMERATOR = 1
HOLDOUT_FRACTION_DENOMINATOR = 5
RISK_BAND_ORDER = ("high", "elevated", "guarded", "low")
BASELINE_STRATEGIES = ("original", "exact-ao")
CANDIDATE_STRATEGY = "ao-risk"
DEFAULT_STRATEGY = "original"

_SCHEDULING_FIELDS = {
    "atomic_numbers",
    "exact_ao_count",
    "molecular_charge",
    "spin_channels",
    "unpaired_electrons",
}
_OUTCOME_FIELD_TOKENS = (
    "converg",
    "iteration",
    "status",
    "success",
    "failure",
    "outcome",
    "result",
    "energy",
    "force",
    "label",
    "target",
)


@dataclass(frozen=True)
class SchedulingCase:
    """Validated identity and explicit input-only metadata for one system."""

    case_id: str
    molecule_group_id: str
    atomic_numbers: tuple[int, ...]
    exact_ao_count: int
    molecular_charge: float
    spin_channels: int | None
    unpaired_electrons: int | None

    def canonical_record(self) -> dict[str, object]:
        """Return the normalized fields used to identify the frozen input set."""
        metadata: dict[str, object] = {
            "atomic_numbers": list(self.atomic_numbers),
            "exact_ao_count": self.exact_ao_count,
            "molecular_charge": self.molecular_charge,
        }
        if self.spin_channels is not None:
            metadata["spin_channels"] = self.spin_channels
        if self.unpaired_electrons is not None:
            metadata["unpaired_electrons"] = self.unpaired_electrons
        return {
            "case_id": self.case_id,
            "molecule_group_id": self.molecule_group_id,
            "scheduling_metadata": metadata,
        }


@dataclass(frozen=True)
class GroupingAnnotation:
    """Explainable risk and AO bucket labels keyed by case ID for a planner."""

    case_id: str
    exact_ao_count: int
    risk_score: int
    risk_band: str
    risk_reasons: tuple[str, ...]

    @property
    def planner_bucket_key(self) -> tuple[int, int]:
        """Return AO-then-risk ordering metadata, without permuting any cases."""
        return self.exact_ao_count, RISK_BAND_ORDER.index(self.risk_band)


def _canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON values with one byte representation and no NaN tokens."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConvergenceGroupingError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc
    return encoded.encode("utf-8")


def _sha256_json(value: object) -> str:
    """Hash a canonical JSON value for the preregistered identities."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def ordered_case_ids_sha256(case_ids: tuple[str, ...] | list[str]) -> str:
    """Hash the complete ordered cohort independently of its grouping permutation.

    The expected identity must be pinned before measurement; a partition check
    alone cannot detect a difficult system removed from the submitted view.
    """
    identities = [_identity(case_id, "cohort case ID") for case_id in case_ids]
    if len(identities) != len(set(identities)):
        raise ConvergenceGroupingError("cohort case IDs must be unique")
    return _sha256_json(identities)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys so parsing cannot silently change a plan."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConvergenceGroupingError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject the non-standard NaN and infinity tokens accepted by json.loads."""
    raise ConvergenceGroupingError(f"non-finite JSON number is forbidden: {value}")


def load_json_document(path: Path) -> object:
    """Load strict JSON while rejecting duplicate keys and non-finite constants."""
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConvergenceGroupingError(
            f"cannot read JSON document {path}: {exc}"
        ) from exc


def _field_token(key: str) -> str:
    return "".join(character for character in key.casefold() if character.isalnum())


def _reject_outcome_fields(value: object, location: str) -> None:
    """Reject outcome-shaped keys before accepting any scheduling metadata."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConvergenceGroupingError(
                    f"{location} object keys must be strings"
                )
            token = _field_token(key)
            if any(forbidden in token for forbidden in _OUTCOME_FIELD_TOKENS):
                raise ConvergenceGroupingError(
                    f"outcome field {key!r} is forbidden in {location}"
                )
            _reject_outcome_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_outcome_fields(child, f"{location}[{index}]")


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], location: str
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ConvergenceGroupingError(f"{location} has " + " and ".join(details))


def _identity(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConvergenceGroupingError(f"{field} must be a nonempty, trimmed string")
    return value


def _finite_charge(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConvergenceGroupingError("molecular_charge must be a finite number")
    charge = float(value)
    if not math.isfinite(charge):
        raise ConvergenceGroupingError("molecular_charge must be a finite number")
    return 0.0 if charge == 0.0 else charge


def _optional_positive_integer(
    metadata: Mapping[str, object], field: str, *, allow_zero: bool = False
) -> int | None:
    if field not in metadata:
        return None
    value = metadata[field]
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ConvergenceGroupingError(f"{field} must be a {qualifier} integer")
    return value


def parse_scheduling_manifest(document: object) -> tuple[SchedulingCase, ...]:
    """Validate an input-only manifest with mandatory molecule-group identity.

    The strict allowlist keeps measured outcomes out of the scheduler contract.
    Case IDs are retained only for lookup and downstream planner tie-breaking;
    molecule-group IDs are used only to keep related conformers in one split.
    """
    if not isinstance(document, Mapping):
        raise ConvergenceGroupingError("scheduling manifest must be a JSON object")
    _reject_outcome_fields(document, "scheduling manifest")
    _require_exact_keys(document, {"schema_version", "systems"}, "scheduling manifest")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
    ):
        raise ConvergenceGroupingError(f"schema_version must equal {SCHEMA_VERSION}")
    systems = document["systems"]
    if not isinstance(systems, list) or not systems:
        raise ConvergenceGroupingError("systems must be a nonempty list")

    cases: list[SchedulingCase] = []
    seen_case_ids: set[str] = set()
    for index, system in enumerate(systems):
        location = f"systems[{index}]"
        if not isinstance(system, Mapping):
            raise ConvergenceGroupingError(f"{location} must be an object")
        _reject_outcome_fields(system, location)
        _require_exact_keys(
            system,
            {"case_id", "molecule_group_id", "scheduling_metadata"},
            location,
        )
        case_id = _identity(system["case_id"], f"{location}.case_id")
        if case_id in seen_case_ids:
            raise ConvergenceGroupingError(
                f"case_id values must be unique: {case_id!r}"
            )
        seen_case_ids.add(case_id)
        group_id = _identity(
            system["molecule_group_id"], f"{location}.molecule_group_id"
        )
        metadata = system["scheduling_metadata"]
        if not isinstance(metadata, Mapping):
            raise ConvergenceGroupingError(
                f"{location}.scheduling_metadata must be an object"
            )
        _reject_outcome_fields(metadata, f"{location}.scheduling_metadata")
        required = {"atomic_numbers", "exact_ao_count", "molecular_charge"}
        missing = required - metadata.keys()
        unknown = metadata.keys() - _SCHEDULING_FIELDS
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if unknown:
                details.append(f"unknown {sorted(unknown)}")
            raise ConvergenceGroupingError(
                f"{location}.scheduling_metadata has " + " and ".join(details)
            )

        atomic_numbers = metadata["atomic_numbers"]
        if (
            not isinstance(atomic_numbers, list)
            or not atomic_numbers
            or any(
                type(number) is not int or not 1 <= number <= 118
                for number in atomic_numbers
            )
        ):
            raise ConvergenceGroupingError(
                f"{location}.scheduling_metadata.atomic_numbers must be a "
                "nonempty list of integers in [1, 118]"
            )
        ao_count = metadata["exact_ao_count"]
        if type(ao_count) is not int or ao_count <= 0:
            raise ConvergenceGroupingError(
                f"{location}.scheduling_metadata.exact_ao_count must be a "
                "positive integer"
            )
        spin_channels = _optional_positive_integer(metadata, "spin_channels")
        unpaired_electrons = _optional_positive_integer(
            metadata, "unpaired_electrons", allow_zero=True
        )
        if spin_channels is None and unpaired_electrons is None:
            raise ConvergenceGroupingError(
                f"{location}.scheduling_metadata requires an explicit "
                "spin_channels or unpaired_electrons field"
            )
        cases.append(
            SchedulingCase(
                case_id=case_id,
                molecule_group_id=group_id,
                atomic_numbers=tuple(atomic_numbers),
                exact_ao_count=ao_count,
                molecular_charge=_finite_charge(metadata["molecular_charge"]),
                spin_channels=spin_channels,
                unpaired_electrons=unpaired_electrons,
            )
        )
    return tuple(cases)


def policy_document() -> dict[str, object]:
    """Return the immutable-by-convention policy contract covered by its hash."""
    return {
        "policy_id": "issue-514-static-ao-risk-v1",
        "candidate_strategy": CANDIDATE_STRATEGY,
        "baseline_strategies": list(BASELINE_STRATEGIES),
        "default_strategy": DEFAULT_STRATEGY,
        "fitting": "none; all risk rules are fixed before outcomes",
        "allowed_features": [
            "atomic_numbers",
            "exact_ao_count",
            "molecular_charge",
            "spin_channels when explicitly supplied",
            "unpaired_electrons when explicitly supplied",
        ],
        "identity_only_fields": ["case_id", "molecule_group_id"],
        "risk_points": [
            {"condition": "molecular_charge != 0", "points": 1},
            {"condition": "explicit unpaired_electrons > 0", "points": 1},
            {"condition": "explicit spin_channels > 1", "points": 1},
            {"condition": "any atomic_number >= 18", "points": 1},
        ],
        "risk_score_range": [0, 4],
        "risk_bands": {
            "low": [0, 0],
            "guarded": [1, 1],
            "elevated": [2, 2],
            "high": [3, 4],
        },
        "planner_bucket_fields": ["exact_ao_count", "risk_band"],
        "planner_bucket_order": ["exact_ao_count ascending", "risk_band high to low"],
        "ordering_and_mapping_owner": "benchmarks.ao_grouping finite-list planner",
        "invalid_or_unknown_scheduling_metadata": "reject",
        "production_or_default_change": False,
        "deployment_claim": False,
    }


def annotate_scheduling_inputs(
    document: object,
) -> dict[str, GroupingAnnotation]:
    """Map each case ID to a frozen AO/risk annotation without reordering cases.

    The returned mapping is suitable for a later planner adapter. It is not a
    batch plan: callers must let the #513 finite-list planner create the one
    complete permutation and its inverse mapping.
    """
    cases = parse_scheduling_manifest(document)
    annotations: dict[str, GroupingAnnotation] = {}
    for case in cases:
        reasons: list[str] = []
        if case.molecular_charge != 0.0:
            reasons.append("nonzero_molecular_charge")
        if case.unpaired_electrons is not None and case.unpaired_electrons > 0:
            reasons.append("explicit_unpaired_electrons")
        if case.spin_channels is not None and case.spin_channels > 1:
            reasons.append("explicit_multiple_spin_channels")
        if max(case.atomic_numbers) >= 18:
            reasons.append("element_Z_at_least_18")
        score = min(4, len(reasons))
        band = (
            "low"
            if score == 0
            else "guarded"
            if score == 1
            else "elevated"
            if score == 2
            else "high"
        )
        annotations[case.case_id] = GroupingAnnotation(
            case_id=case.case_id,
            exact_ao_count=case.exact_ao_count,
            risk_score=score,
            risk_band=band,
            risk_reasons=tuple(reasons),
        )
    return {case_id: annotations[case_id] for case_id in sorted(annotations)}


def _split_document(cases: tuple[SchedulingCase, ...]) -> dict[str, object]:
    by_group: dict[str, list[str]] = {}
    for case in cases:
        by_group.setdefault(case.molecule_group_id, []).append(case.case_id)
    groups = sorted(by_group)
    if len(groups) < 2:
        raise ConvergenceGroupingError(
            "at least two molecule_group_id values are required for an "
            "independent holdout"
        )
    holdout_count = math.ceil(
        len(groups) * HOLDOUT_FRACTION_NUMERATOR / HOLDOUT_FRACTION_DENOMINATOR
    )
    holdout_count = min(max(1, holdout_count), len(groups) - 1)
    ranked_groups = sorted(
        groups,
        key=lambda group_id: (
            hashlib.sha256(f"{SPLIT_SEED}\0{group_id}".encode()).hexdigest(),
            group_id,
        ),
    )
    holdout_groups = set(ranked_groups[-holdout_count:])
    return {
        "algorithm": "sha256-group-rank-v1",
        "seed": SPLIT_SEED,
        "holdout_fraction": {
            "numerator": HOLDOUT_FRACTION_NUMERATOR,
            "denominator": HOLDOUT_FRACTION_DENOMINATOR,
        },
        "holdout_group_count": holdout_count,
        "groups": [
            {
                "molecule_group_id": group_id,
                "rank_sha256": hashlib.sha256(
                    f"{SPLIT_SEED}\0{group_id}".encode()
                ).hexdigest(),
                "partition": "holdout" if group_id in holdout_groups else "calibration",
                "case_ids": sorted(by_group[group_id]),
            }
            for group_id in groups
        ],
    }


def build_freeze_plan(document: object) -> dict[str, object]:
    """Create policy, split, and input identity hashes before any outcomes."""
    cases = parse_scheduling_manifest(document)
    normalized_systems = [
        case.canonical_record() for case in sorted(cases, key=lambda item: item.case_id)
    ]
    input_identity = {
        "schema_version": SCHEMA_VERSION,
        "systems": normalized_systems,
    }
    policy = policy_document()
    split = _split_document(cases)
    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "preregistered_only",
        "policy": policy,
        "policy_sha256": _sha256_json(policy),
        "split": split,
        "split_sha256": _sha256_json(split),
        "input_identity_sha256": _sha256_json(input_identity),
        "input_system_count": len(cases),
        "input_group_count": len(split["groups"]),
        "comparison_strategies": ["original", "exact-ao", "ao-risk"],
        "evaluation_protocol": evaluation_protocol_document(),
    }
    plan["freeze_plan_sha256"] = _sha256_json(plan)
    return plan


def validate_freeze_plan(
    document: object,
    frozen_plan: object,
    *,
    expected_freeze_plan_sha256: str | None = None,
) -> tuple[SchedulingCase, ...]:
    """Bind input-only metadata to the current policy and externally pinned freeze.

    Self-consistent hashes alone do not prevent a rewritten policy or split.
    Reconstructing the entire preregistration also rejects changed thresholds,
    partitions and numeric-type aliases. An expected identity must come from
    the experiment's prior checkpoint, not be rediscovered from this file.
    """
    if not isinstance(frozen_plan, Mapping):
        raise ConvergenceGroupingError("frozen plan must be an object")
    if expected_freeze_plan_sha256 is not None:
        if (
            not isinstance(expected_freeze_plan_sha256, str)
            or len(expected_freeze_plan_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_freeze_plan_sha256
            )
        ):
            raise ConvergenceGroupingError(
                "expected freeze identity must be a lowercase SHA-256"
            )
        if frozen_plan.get("freeze_plan_sha256") != expected_freeze_plan_sha256:
            raise ConvergenceGroupingError(
                "frozen plan does not match the expected experiment identity"
            )
    expected = build_freeze_plan(document)
    if _sha256_json(frozen_plan) != _sha256_json(expected):
        raise ConvergenceGroupingError(
            "frozen plan does not match its input metadata, current policy "
            "and preregistered split"
        )
    return parse_scheduling_manifest(document)


def evaluation_protocol_document() -> dict[str, object]:
    """Return the prospective timing, correctness, and decision matrix."""
    return {
        "scope": "finite-list FRESH xTBloom CUDA public calls with host descriptors",
        "same_for_all_strategies": [
            "clean source and identical library binary",
            "same input systems and system order before planner permutation",
            "same model, numerical tolerances, SCC controls, and requested outputs",
            "strict FRESH start and fixed device/thread settings",
        ],
        "strategies": ["original", "exact-ao", "ao-risk"],
        "batch_sizes": [64, 256],
        "cohorts": [
            "all four frozen OMol25 views (2048 systems total)",
            "first256 frozen view",
            "last256 frozen view",
            "homogeneous 32-atom control",
            "homogeneous 62-atom control",
        ],
        "warmup_runs_per_strategy_and_coordinate": 5,
        "paired_measurement_blocks_per_coordinate": 30,
        "interleaving": "cycle all six strategy permutations five times per coordinate",
        "primary_speedup": "exact-ao end-to-end time divided by ao-risk "
        "end-to-end time",
        "bootstrap": {
            "method": "paired percentile bootstrap over measurement blocks",
            "replicates": 10000,
            "seed": "xtbloom-issue-514-bootstrap-v1",
        },
        "reported_time_distributions": [
            "median",
            "p95",
            "maximum",
            "paired 95% bootstrap interval",
        ],
        "timed_stages_ms": [
            "feature annotation and planning",
            "public preparation",
            "compute",
            "publication and result scatter",
            "complete end-to-end call including planning",
            "per-batch time including the final tail batch",
        ],
        "other_costs": ["plan construction memory", "peak process memory"],
        "paired_correctness_per_system": [
            "energy, forces, and charges against the pinned independent reference",
            "status, convergence, SCC iterations, and complete failed-system "
            "output slices",
            "success set and failure reason without filtering any requested system",
        ],
        "denominator": "every case_id in the frozen input manifest, including "
        "failures and unavailable results",
        "accept_for_scoped_opt_in_followup": [
            "all expected IDs remain in every strategy and correctness gates pass",
            "ao-risk has at least 5% median end-to-end improvement over exact-ao "
            "on a predeclared eligible holdout cohort",
            "paired 95% bootstrap interval for that cohort's end-to-end speedup "
            "excludes 1.0",
            "no other eligible cohort has more than 5% median end-to-end regression",
        ],
        "no_go": [
            "any input is dropped, duplicated, or unreported",
            "any scientific correctness or status regression is attributable to "
            "grouping",
            "feature/planning overhead removes the end-to-end gain",
            "completed eligible evaluation does not meet the acceptance rule",
        ],
        "unverified_or_blocked": [
            "required input, independent oracle, GPU, or holdout evidence is "
            "missing or unavailable",
            "evaluation is incomplete; missing prerequisites are not an "
            "empirical policy result and do not permit issue closure",
        ],
        "decision_scope": "accept only an opt-in follow-up for named eligible "
        "cohorts; original remains default",
        "no_result_claim": "this frozen plan contains no training result or "
        "holdout measurement",
    }


def _validated_split_groups(
    frozen_plan: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    """Verify frozen hashes and return the group partition and case mapping."""
    if not isinstance(frozen_plan, Mapping):
        raise ConvergenceGroupingError("frozen plan must be an object")
    split = frozen_plan.get("split")
    if not isinstance(split, Mapping) or _sha256_json(split) != frozen_plan.get(
        "split_sha256"
    ):
        raise ConvergenceGroupingError(
            "frozen split hash does not match its serialized split"
        )
    policy = frozen_plan.get("policy")
    if not isinstance(policy, Mapping) or _sha256_json(policy) != frozen_plan.get(
        "policy_sha256"
    ):
        raise ConvergenceGroupingError(
            "frozen policy hash does not match its serialized policy"
        )
    plan_without_hash = {
        key: value for key, value in frozen_plan.items() if key != "freeze_plan_sha256"
    }
    if _sha256_json(plan_without_hash) != frozen_plan.get("freeze_plan_sha256"):
        raise ConvergenceGroupingError(
            "freeze-plan hash does not match the serialized plan"
        )
    groups = split.get("groups")
    if not isinstance(groups, list):
        raise ConvergenceGroupingError("frozen split groups must be a list")
    partitions: dict[str, str] = {}
    case_groups: dict[str, str] = {}
    for entry in groups:
        if not isinstance(entry, Mapping):
            raise ConvergenceGroupingError("frozen split group entries must be objects")
        _require_exact_keys(
            entry,
            {"molecule_group_id", "rank_sha256", "partition", "case_ids"},
            "frozen split group",
        )
        group_id = _identity(entry["molecule_group_id"], "frozen molecule_group_id")
        partition = entry["partition"]
        if partition not in {"calibration", "holdout"} or group_id in partitions:
            raise ConvergenceGroupingError(
                "frozen split has an invalid or duplicate group"
            )
        partitions[group_id] = partition
        case_ids = entry["case_ids"]
        if not isinstance(case_ids, list) or not case_ids:
            raise ConvergenceGroupingError(
                "frozen split group case_ids must be nonempty"
            )
        for case_value in case_ids:
            case_id = _identity(case_value, "frozen case_id")
            if case_id in case_groups:
                raise ConvergenceGroupingError("frozen split repeats a case_id")
            case_groups[case_id] = group_id
    if not partitions or not any(
        value == "calibration" for value in partitions.values()
    ):
        raise ConvergenceGroupingError("frozen split has no calibration groups")
    if not any(value == "holdout" for value in partitions.values()):
        raise ConvergenceGroupingError("frozen split has no holdout groups")
    return partitions, case_groups


def validate_calibration_input(
    calibration_document: object, frozen_plan: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    """Validate a separate labeled calibration file and reject every holdout ID.

    SCC iteration, convergence, and status outcomes are legal only in this
    purpose-tagged calibration document; scheduling metadata never accepts
    them. This prototype does not fit a model, but the guard makes later
    calibration work respect the frozen split.
    """
    if not isinstance(calibration_document, Mapping):
        raise ConvergenceGroupingError("calibration input must be an object")
    _require_exact_keys(
        calibration_document,
        {"schema_version", "purpose", "records"},
        "calibration input",
    )
    if (
        type(calibration_document["schema_version"]) is not int
        or calibration_document["schema_version"] != SCHEMA_VERSION
        or calibration_document["purpose"] != "calibration-only"
    ):
        raise ConvergenceGroupingError(
            "calibration input must declare schema v1 and purpose calibration-only"
        )
    partitions, case_groups = _validated_split_groups(frozen_plan)
    records = calibration_document["records"]
    if not isinstance(records, list) or not records:
        raise ConvergenceGroupingError("calibration records must be a nonempty list")
    normalized: list[dict[str, object]] = []
    seen_cases: set[str] = set()
    for index, record in enumerate(records):
        location = f"calibration records[{index}]"
        if not isinstance(record, Mapping):
            raise ConvergenceGroupingError(f"{location} must be an object")
        _require_exact_keys(
            record,
            {
                "case_id",
                "molecule_group_id",
                "scc_iterations",
                "scc_converged",
                "status",
            },
            location,
        )
        case_id = _identity(record["case_id"], f"{location}.case_id")
        group_id = _identity(
            record["molecule_group_id"], f"{location}.molecule_group_id"
        )
        if case_id in seen_cases:
            raise ConvergenceGroupingError(
                f"calibration input repeats case_id {case_id!r}"
            )
        seen_cases.add(case_id)
        if case_groups.get(case_id) != group_id:
            raise ConvergenceGroupingError(
                f"calibration case_id {case_id!r} does not match the frozen "
                "group mapping"
            )
        partition = partitions.get(group_id)
        if partition == "holdout":
            raise ConvergenceGroupingError(
                "calibration input cannot contain holdout molecule_group_id "
                f"{group_id!r}"
            )
        if partition != "calibration":
            raise ConvergenceGroupingError(
                f"calibration input contains unknown molecule_group_id {group_id!r}"
            )
        iterations = record["scc_iterations"]
        if type(iterations) is not int or iterations < 0:
            raise ConvergenceGroupingError(
                f"{location}.scc_iterations must be a nonnegative integer"
            )
        converged = record["scc_converged"]
        if type(converged) is not bool:
            raise ConvergenceGroupingError(f"{location}.scc_converged must be boolean")
        status = record["status"]
        if type(status) is not int:
            raise ConvergenceGroupingError(f"{location}.status must be an integer")
        normalized.append(
            {
                "case_id": case_id,
                "molecule_group_id": group_id,
                "scc_iterations": iterations,
                "scc_converged": converged,
                "status": status,
            }
        )
    return tuple(sorted(normalized, key=lambda item: item["case_id"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze-plan", help="serialize policy, group split, and input identity hashes"
    )
    freeze.add_argument("--manifest", required=True, type=Path)
    freeze.add_argument(
        "--output",
        required=True,
        help="new output path, or '-' for standard output; existing files "
        "are preserved",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the preregistration-only command-line interface."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        plan = build_freeze_plan(load_json_document(arguments.manifest))
        serialized = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if arguments.output == "-":
            sys.stdout.write(serialized)
        else:
            output = Path(arguments.output)
            if output.exists():
                raise ConvergenceGroupingError(
                    f"refusing to overwrite existing freeze plan: {output}"
                )
            output.write_text(serialized, encoding="utf-8")
    except (ConvergenceGroupingError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
