#!/usr/bin/env python3
"""Analyze P0-B equivalence and Experiment 3-A convergence JSONL.

The analyzer preserves every failure category.  Numerical errors use only the
paired-success subset; convergence denominators use every eligible row.
OMol25 stress rows are reported separately and never enter the weighted main
estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any


class AnalysisError(RuntimeError):
    pass


def iter_records(root: Path) -> Iterable[dict[str, Any]]:
    paths = [root] if root.is_file() else sorted(root.rglob("results.jsonl"))
    if not paths:
        raise AnalysisError(f"no results.jsonl found under {root}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise AnalysisError(f"{path}:{line_number}: {exc}") from exc


def available(field: dict[str, Any]) -> Any | None:
    return field.get("value") if field.get("availability") == "available" else None


def identity(record: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    inp, program = record["input"], record["program"]
    return (
        inp["dataset"],
        inp.get("subset", "main"),
        inp["system_id"],
        program["engine"],
        program.get("backend", "cpu"),
        program.get("memory_mode", "host"),
    )


def flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        result: list[float] = []
        for value in values:
            result.extend(flatten(value))
        return result
    return [float(values)]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return (
        ordered[lo]
        if lo == hi
        else ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
    )


def weighted_percentile(
    values: list[float], weights: list[float], q: float
) -> float | None:
    """Return the inverse-probability weighted empirical quantile.

    The ordinary quantiles remain useful descriptions of the realized sample,
    while this separate estimand represents the frozen probability sample's
    target population.  Stress samples deliberately never call this helper.
    """
    if not values:
        return None
    if len(values) != len(weights) or any(weight <= 0.0 for weight in weights):
        raise AnalysisError("weighted quantiles require one positive weight per value")
    ordered = sorted(zip(values, weights, strict=True))
    target = q * sum(weights)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def distribution(
    values: list[float], weights: list[float] | None = None
) -> dict[str, Any]:
    document = {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values) if values else None,
        "quantile_estimand": "unweighted realized sample",
    }
    document.update(
        {
            "weighted_population_p50": weighted_percentile(values, weights, 0.50)
            if weights
            else None,
            "weighted_population_p95": weighted_percentile(values, weights, 0.95)
            if weights
            else None,
            "weighted_population_p99": weighted_percentile(values, weights, 0.99)
            if weights
            else None,
            "weighted_quantile_estimand": (
                "inverse-probability weighted target population" if weights else None
            ),
        }
    )
    return document


def wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [center - half, center + half]


def sampling_weight(record: dict[str, Any]) -> float:
    if record["input"].get("subset") == "stress":
        # The stress set is a deliberately enriched diagnostic sample, not a
        # probability sample and never contributes to a population estimate.
        return 1.0
    value = record["input"].get("strata", {}).get("sampling_probability")
    if value in (None, ""):
        raise AnalysisError(
            f"missing sampling_probability for {record['input']['system_id']}"
        )
    probability = float(value)
    if not 0.0 < probability <= 1.0:
        raise AnalysisError(
            f"invalid sampling_probability for {record['input']['system_id']}: {value}"
        )
    return 1.0 / probability


def weighted_rate(rows: list[dict[str, Any]], predicate: Any) -> float | None:
    if not rows:
        return None
    weights = [sampling_weight(row) for row in rows]
    return sum(
        weight * bool(predicate(row)) for weight, row in zip(weights, rows, strict=True)
    ) / sum(weights)


def weighted_kish_wilson_interval(
    rows: list[dict[str, Any]], predicate: Any
) -> list[float] | None:
    """Approximate a weighted-rate interval using Kish effective sample size.

    This is labelled separately from the exact unweighted binomial Wilson
    interval so the two estimands cannot be confused in paper tables.
    """
    if not rows:
        return None
    weights = [sampling_weight(row) for row in rows]
    rate = sum(
        weight * bool(predicate(row)) for weight, row in zip(weights, rows, strict=True)
    ) / sum(weights)
    effective_n = sum(weights) ** 2 / sum(weight * weight for weight in weights)
    z = 1.959963984540054
    denominator = 1 + z * z / effective_n
    center = (rate + z * z / (2 * effective_n)) / denominator
    half = (
        z
        * math.sqrt(
            rate * (1 - rate) / effective_n + z * z / (4 * effective_n * effective_n)
        )
        / denominator
    )
    return [center - half, center + half]


def ecdf(values: list[float]) -> list[dict[str, float | int]]:
    counts = Counter(values)
    total = sum(counts.values())
    cumulative = 0
    output = []
    for value in sorted(counts):
        cumulative += counts[value]
        output.append(
            {"iteration": value, "count": counts[value], "fraction": cumulative / total}
        )
    return output


def success(record: dict[str, Any]) -> bool:
    return record["status"]["category"] == "success"


def eligible(record: dict[str, Any]) -> bool:
    return record["status"]["category"] not in {
        "unsupported",
        "malformed",
        "invalid_input",
    }


def compare(
    lhs: dict[str, Any], rhs: dict[str, Any], gates: dict[str, float]
) -> dict[str, Any]:
    row: dict[str, Any] = {"paired_success": success(lhs) and success(rhs)}
    if not row["paired_success"]:
        return row
    for name, gate_name in (
        ("energy", "energy"),
        ("forces", "force"),
        ("atomic_charges", "charge"),
    ):
        left = available(lhs["results"][name])
        right = available(rhs["results"][name])
        if left is None or right is None:
            row[name] = {"availability": "unavailable"}
            continue
        a, b = flatten(left), flatten(right)
        if len(a) != len(b):
            raise AnalysisError(f"shape mismatch for {name}")
        if not a or not all(math.isfinite(value) for value in (*a, *b)):
            row[name] = {
                "availability": "available",
                "maximum": None,
                "rmse": None,
                "passed": False,
                "reason": "empty or non-finite value",
            }
            continue
        errors = [abs(x - y) for x, y in zip(a, b, strict=True)]
        maximum = max(errors, default=0.0)
        rmse = math.sqrt(sum(value * value for value in errors) / max(1, len(errors)))
        metrics: dict[str, Any] = {
            "availability": "available",
            "maximum": maximum,
            "rmse": rmse,
            "passed": maximum <= gates[gate_name],
        }
        atom_count = len(lhs.get("input", {}).get("atomic_numbers", ()))
        if name == "energy" and atom_count:
            metrics["absolute_error_per_atom"] = maximum / atom_count
        elif name == "forces" and errors and len(errors) % 3 == 0:
            metrics["atom_vector_rmse"] = math.sqrt(
                sum(
                    sum(
                        component * component for component in errors[index : index + 3]
                    )
                    for index in range(0, len(errors), 3)
                )
                / (len(errors) // 3)
            )
            metrics["component_rmse"] = rmse
        elif name == "atomic_charges":
            metrics["charge_rmse"] = rmse
        row[name] = metrics
    return row


def metric_distribution(
    rows: list[dict[str, Any]],
    quantity: str,
    metric: str,
    use_population_weights: bool,
) -> dict[str, Any]:
    present = [row for row in rows if row[quantity].get(metric) is not None]
    values = [float(row[quantity][metric]) for row in present]
    weights = (
        [float(row["weight"]) for row in present] if use_population_weights else None
    )
    return distribution(values, weights)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--outliers-csv", type=Path, required=True)
    parser.add_argument("--paired-errors-jsonl", type=Path, required=True)
    parser.add_argument("--energy-atol", type=float, default=5e-7)
    parser.add_argument("--force-atol", type=float, default=5e-6)
    parser.add_argument("--charge-atol", type=float, default=5e-7)
    parser.add_argument("--cpu-cuda-atol", type=float, default=1e-6)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--require-complete-p0", action="store_true")
    parser.add_argument("--require-dxtb", action="store_true")
    args = parser.parse_args()
    hard_limits = {"energy": 5e-7, "force": 5e-6, "charge": 5e-7}
    requested = {
        "energy": args.energy_atol,
        "force": args.force_atol,
        "charge": args.charge_atol,
    }
    if any(requested[key] > hard_limits[key] for key in hard_limits):
        raise AnalysisError("requested tolerance widens the paper contract")
    if args.cpu_cuda_atol > 1e-6:
        raise AnalysisError("requested CPU-CUDA tolerance widens the paper contract")
    if args.require_complete_p0 and args.selection is None:
        raise AnalysisError("--require-complete-p0 requires --selection")

    records: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for record in iter_records(args.input_root):
        key = identity(record)
        if key in records:
            raise AnalysisError(f"duplicate result coordinate: {key}")
        records[key] = record

    by_system: dict[
        tuple[str, str, str], dict[tuple[str, str, str], dict[str, Any]]
    ] = defaultdict(dict)
    for (
        dataset,
        subset,
        system_id,
        engine,
        backend,
        memory,
    ), record in records.items():
        by_system[(dataset, subset, system_id)][(engine, backend, memory)] = record

    cpu_cuda_gates = {
        "energy": args.cpu_cuda_atol,
        "force": args.cpu_cuda_atol,
        "charge": args.cpu_cuda_atol,
    }
    # label, left, right, tolerance set, quantities required on paired successes,
    # formal completeness requirement, and whether numerical outliers block P0.
    comparisons = [
        (
            "xtbloom-cpu-vs-xtb",
            ("xtbloom", "cpu", "host"),
            ("xtb", "cpu", "host"),
            requested,
            ("energy", "forces"),
            True,
            True,
        ),
        (
            "xtbloom-cpu-vs-tblite",
            ("xtbloom", "cpu", "host"),
            ("tblite", "cpu", "host"),
            requested,
            ("energy", "forces", "atomic_charges"),
            True,
            True,
        ),
        (
            "xtbloom-cuda-host-vs-xtb",
            ("xtbloom", "cuda", "host"),
            ("xtb", "cpu", "host"),
            requested,
            ("energy", "forces"),
            True,
            True,
        ),
        (
            "xtbloom-cuda-device-vs-xtb",
            ("xtbloom", "cuda", "device"),
            ("xtb", "cpu", "host"),
            requested,
            ("energy", "forces"),
            True,
            True,
        ),
        (
            "xtbloom-cuda-mixed-vs-xtb",
            ("xtbloom", "cuda", "mixed"),
            ("xtb", "cpu", "host"),
            requested,
            ("energy", "forces"),
            True,
            True,
        ),
        (
            "xtbloom-cpu-vs-cuda-host",
            ("xtbloom", "cpu", "host"),
            ("xtbloom", "cuda", "host"),
            cpu_cuda_gates,
            ("energy", "forces", "atomic_charges"),
            True,
            True,
        ),
        (
            "xtbloom-cpu-vs-cuda-device",
            ("xtbloom", "cpu", "host"),
            ("xtbloom", "cuda", "device"),
            cpu_cuda_gates,
            ("energy", "forces", "atomic_charges"),
            True,
            True,
        ),
        (
            "xtbloom-cpu-vs-cuda-mixed",
            ("xtbloom", "cpu", "host"),
            ("xtbloom", "cuda", "mixed"),
            cpu_cuda_gates,
            ("energy", "forces", "atomic_charges"),
            True,
            True,
        ),
        (
            "xtbloom-cpu-vs-dxtb",
            ("xtbloom", "cpu", "host"),
            ("dxtb", "cpu", "host"),
            requested,
            ("energy", "forces"),
            args.require_dxtb,
            False,
        ),
        (
            "xtbloom-cuda-vs-dxtb",
            ("xtbloom", "cuda", "device"),
            ("dxtb", "cuda", "device"),
            requested,
            ("energy", "forces"),
            args.require_dxtb,
            False,
        ),
    ]
    expected_groups = {("qm9", "main"), ("omol25", "main"), ("omol25", "stress")}
    expected_systems: dict[tuple[str, str], set[str]] = {}
    if args.selection is not None:
        if not args.selection.is_file():
            raise AnalysisError(f"frozen selection is missing: {args.selection}")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        inventories = selection.get("inventories", {})
        sources = {
            ("qm9", "main"): ("qm9_main", "main"),
            ("omol25", "main"): ("omol25", "main"),
            ("omol25", "stress"): ("omol25", "stress"),
        }
        for group, (inventory_name, subset_name) in sources.items():
            values = (
                inventories.get(inventory_name, {})
                .get("system_ids_by_subset", {})
                .get(subset_name)
            )
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) for value in values)
            ):
                raise AnalysisError(
                    f"frozen selection lacks exact ID universe for {group}"
                )
            if len(values) != len(set(values)):
                raise AnalysisError(
                    f"frozen selection contains duplicate IDs for {group}"
                )
            expected_systems[group] = set(values)
    base_systems: dict[tuple[str, str], set[str]] = defaultdict(set)
    for coordinate, engines in by_system.items():
        dataset, subset, system_id = coordinate
        if ("xtbloom", "cpu", "host") in engines:
            base_systems[(dataset, subset)].add(system_id)
    completeness_failures: list[dict[str, Any]] = []
    if args.require_complete_p0:
        for dataset, subset in sorted(expected_groups):
            expected = expected_systems[(dataset, subset)]
            actual = base_systems[(dataset, subset)]
            missing_ids = sorted(expected - actual)
            extra_ids = sorted(actual - expected)
            if missing_ids or extra_ids:
                completeness_failures.append(
                    {
                        "comparison": "matrix",
                        "dataset": dataset,
                        "subset": subset,
                        "reason": "CPU xTBloom system universe differs from frozen selection",
                        "expected_count": len(expected),
                        "actual_count": len(actual),
                        "missing_count": len(missing_ids),
                        "extra_count": len(extra_ids),
                        "missing_ids_sample": missing_ids[:16],
                        "extra_ids_sample": extra_ids[:16],
                    }
                )
    equivalence: dict[str, Any] = {}
    outliers: list[dict[str, Any]] = []
    diagnostic_outliers: list[dict[str, Any]] = []
    paired_error_ledger: list[dict[str, Any]] = []
    for (
        label,
        left_key,
        right_key,
        gates,
        required_quantities,
        required,
        blocking,
    ) in comparisons:
        grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        groups = (
            expected_groups
            if args.require_complete_p0 and required
            else set(base_systems)
        )
        for dataset, subset in sorted(groups):
            universe = (
                expected_systems[(dataset, subset)]
                if args.require_complete_p0 and required
                else base_systems[(dataset, subset)]
            )
            for system_id in sorted(universe):
                engines = by_system[(dataset, subset, system_id)]
                if left_key not in engines or right_key not in engines:
                    if required and args.require_complete_p0:
                        completeness_failures.append(
                            {
                                "comparison": label,
                                "dataset": dataset,
                                "subset": subset,
                                "system_id": system_id,
                                "reason": f"missing coordinate: left={left_key in engines} right={right_key in engines}",
                            }
                        )
                    continue
                result = compare(engines[left_key], engines[right_key], gates)
                result.update(
                    {
                        "dataset": dataset,
                        "subset": subset,
                        "system_id": system_id,
                        "weight": sampling_weight(engines[left_key]),
                        "strata": dict(engines[left_key]["input"].get("strata", {})),
                    }
                )
                grouped_rows[(dataset, subset)].append(result)
                if result["paired_success"]:
                    paired_error_ledger.append(
                        {
                            "comparison": label,
                            "dataset": dataset,
                            "subset": subset,
                            "system_id": system_id,
                            "sampling_weight": result["weight"]
                            if subset != "stress"
                            else None,
                            "strata": result["strata"],
                            "quantities": {
                                quantity: result.get(
                                    quantity, {"availability": "unavailable"}
                                )
                                for quantity in ("energy", "forces", "atomic_charges")
                            },
                        }
                    )
                if result["paired_success"]:
                    for quantity in required_quantities:
                        field = result.get(quantity, {})
                        if field.get("availability") != "available" or not field.get(
                            "passed", False
                        ):
                            target = outliers if blocking else diagnostic_outliers
                            target.append(
                                {
                                    "comparison": label,
                                    "dataset": dataset,
                                    "subset": subset,
                                    "system_id": system_id,
                                    "quantity": quantity,
                                    "maximum": field.get("maximum"),
                                    "reason": field.get(
                                        "reason", "missing required quantity"
                                    ),
                                    "severity": "blocking"
                                    if blocking
                                    else "diagnostic",
                                }
                            )
        if required and blocking and args.require_complete_p0:
            for dataset, subset in sorted(expected_groups):
                rows = grouped_rows.get((dataset, subset), [])
                if not rows or not any(row["paired_success"] for row in rows):
                    completeness_failures.append(
                        {
                            "comparison": label,
                            "dataset": dataset,
                            "subset": subset,
                            "reason": "no paired-success rows",
                        }
                    )
        for (dataset, subset), rows in sorted(grouped_rows.items()):
            quantities = {}
            for quantity in ("energy", "forces", "atomic_charges"):
                available_rows = [
                    row
                    for row in rows
                    if row.get(quantity, {}).get("availability") == "available"
                ]
                weights = [row["weight"] for row in available_rows]
                summary = {
                    **distribution(
                        [row[quantity]["maximum"] for row in available_rows],
                        weights if subset != "stress" else None,
                    ),
                    "gate": gates[
                        {
                            "energy": "energy",
                            "forces": "force",
                            "atomic_charges": "charge",
                        }[quantity]
                    ],
                    "pass_rate": sum(row[quantity]["passed"] for row in available_rows)
                    / len(available_rows)
                    if available_rows
                    else None,
                    "weighted_pass_rate": (
                        sum(
                            weight * row[quantity]["passed"]
                            for weight, row in zip(weights, available_rows, strict=True)
                        )
                        / sum(weights)
                        if weights and subset != "stress"
                        else None
                    ),
                }
                for metric in (
                    "rmse",
                    "absolute_error_per_atom",
                    "component_rmse",
                    "atom_vector_rmse",
                    "charge_rmse",
                ):
                    metric_values = metric_distribution(
                        available_rows, quantity, metric, subset != "stress"
                    )
                    if metric_values["count"]:
                        for name, value in metric_values.items():
                            summary[f"{metric}_{name}"] = value
                quantities[quantity] = summary
            equivalence[f"{label}/{dataset}/{subset}"] = {
                "comparison": label,
                "dataset": dataset,
                "subset": subset,
                "probability_estimate_eligible": subset != "stress",
                "systems": len(rows),
                "paired_success": sum(row["paired_success"] for row in rows),
                "quantities": quantities,
            }

    equivalence_by_stratum: dict[str, Any] = {}
    for coordinate, document in equivalence.items():
        label = document["comparison"]
        dataset = document["dataset"]
        subset = document["subset"]
        # Reconstruct only this comparison's paired rows from by-system so the
        # stratum ledger cannot accidentally reuse the previous loop value.
        comparison = next(item for item in comparisons if item[0] == label)
        left_key, right_key, gates = comparison[1], comparison[2], comparison[3]
        stratum_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        universe = expected_systems.get(
            (dataset, subset), base_systems.get((dataset, subset), set())
        )
        for system_id in sorted(universe):
            engines = by_system[(dataset, subset, system_id)]
            if left_key not in engines or right_key not in engines:
                continue
            result = compare(engines[left_key], engines[right_key], gates)
            result["weight"] = sampling_weight(engines[left_key])
            for name, value in sorted(
                engines[left_key]["input"].get("strata", {}).items()
            ):
                if name == "sampling_probability":
                    continue
                stratum_rows[f"{name}={value}"].append(result)
        for stratum, members in sorted(stratum_rows.items()):
            quantities = {}
            for quantity in ("energy", "forces", "atomic_charges"):
                available_rows = [
                    row
                    for row in members
                    if row.get(quantity, {}).get("availability") == "available"
                ]
                weights = [row["weight"] for row in available_rows]
                summary = {
                    **distribution(
                        [row[quantity]["maximum"] for row in available_rows],
                        weights if subset != "stress" else None,
                    ),
                    "gate": gates[
                        {
                            "energy": "energy",
                            "forces": "force",
                            "atomic_charges": "charge",
                        }[quantity]
                    ],
                    "pass_rate": (
                        sum(row[quantity]["passed"] for row in available_rows)
                        / len(available_rows)
                        if available_rows
                        else None
                    ),
                    "weighted_pass_rate": (
                        sum(
                            weight * row[quantity]["passed"]
                            for weight, row in zip(weights, available_rows, strict=True)
                        )
                        / sum(weights)
                        if weights and subset != "stress"
                        else None
                    ),
                }
                for metric in (
                    "rmse",
                    "absolute_error_per_atom",
                    "component_rmse",
                    "atom_vector_rmse",
                    "charge_rmse",
                ):
                    metric_values = metric_distribution(
                        available_rows, quantity, metric, subset != "stress"
                    )
                    if metric_values["count"]:
                        for name, value in metric_values.items():
                            summary[f"{metric}_{name}"] = value
                quantities[quantity] = summary
            equivalence_by_stratum[f"{coordinate}/{stratum}"] = {
                "comparison": label,
                "dataset": dataset,
                "subset": subset,
                "stratum": stratum,
                "probability_estimate_eligible": subset != "stress",
                "systems": len(members),
                "quantities": quantities,
            }

    convergence: dict[str, Any] = {}
    convergence_by_stratum: dict[str, Any] = {}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (_, subset, _, engine, backend, memory), record in records.items():
        grouped[
            (record["input"]["dataset"], subset, engine, f"{backend}/{memory}")
        ].append(record)
    for key, rows in sorted(grouped.items()):
        denominator = [row for row in rows if eligible(row)]
        successes = sum(success(row) for row in denominator)
        categories = Counter(row["status"]["category"] for row in rows)
        iterations = [
            available(row["results"]["scc_iterations"]) for row in denominator
        ]
        values = [float(value) for value in iterations if value is not None]
        convergence["/".join(key)] = {
            "dataset": key[0],
            "subset": key[1],
            "engine": key[2],
            "execution": key[3],
            "input_rows": len(rows),
            "eligible": len(denominator),
            "successes": successes,
            "rate": successes / len(denominator) if denominator else None,
            "weighted_rate": weighted_rate(denominator, success)
            if key[1] != "stress"
            else None,
            "weighted_kish_wilson_95_approx": weighted_kish_wilson_interval(
                denominator, success
            )
            if key[1] != "stress"
            else None,
            "wilson_95_unweighted_sample_rate": wilson(successes, len(denominator)),
            "categories": dict(sorted(categories.items())),
            "scc_iterations": {
                "median": median(values) if values else None,
                "p95": percentile(values, 0.95),
                "ecdf": ecdf(values),
            },
            "probability_estimate_eligible": key[1] != "stress",
        }

        strata_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for stratum, value in sorted(row["input"].get("strata", {}).items()):
                if stratum == "sampling_probability":
                    continue
                strata_groups[f"{stratum}={value}"].append(row)
        for stratum, members in strata_groups.items():
            denominator = [row for row in members if eligible(row)]
            successes = sum(success(row) for row in denominator)
            iterations = [
                available(row["results"]["scc_iterations"]) for row in denominator
            ]
            iteration_values = [
                float(value) for value in iterations if value is not None
            ]
            convergence_by_stratum["/".join((*key, stratum))] = {
                "dataset": key[0],
                "subset": key[1],
                "engine": key[2],
                "execution": key[3],
                "stratum": stratum,
                "eligible": len(denominator),
                "successes": successes,
                "rate": successes / len(denominator) if denominator else None,
                "weighted_rate": weighted_rate(denominator, success)
                if key[1] != "stress"
                else None,
                "weighted_kish_wilson_95_approx": weighted_kish_wilson_interval(
                    denominator, success
                )
                if key[1] != "stress"
                else None,
                "wilson_95_unweighted_sample_rate": wilson(successes, len(denominator)),
                "categories": dict(
                    sorted(
                        Counter(row["status"]["category"] for row in members).items()
                    )
                ),
                "scc_iterations": {
                    "median": median(iteration_values) if iteration_values else None,
                    "p95": percentile(iteration_values, 0.95),
                    "ecdf": ecdf(iteration_values),
                },
                "probability_estimate_eligible": key[1] != "stress",
            }

    paired_convergence: dict[str, Any] = {}
    for label, left_key, right_key, _, _, _, _ in comparisons:
        tables: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        weighted_tables: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for (dataset, subset, _), engines in by_system.items():
            if left_key not in engines or right_key not in engines:
                continue
            left, right = engines[left_key], engines[right_key]
            if not (eligible(left) and eligible(right)):
                continue
            left_ok, right_ok = success(left), success(right)
            outcome = (
                "both"
                if left_ok and right_ok
                else "left_only"
                if left_ok
                else "right_only"
                if right_ok
                else "neither"
            )
            tables[(dataset, subset)][outcome] += 1
            if subset != "stress":
                left_weight = sampling_weight(left)
                right_weight = sampling_weight(right)
                if not math.isclose(
                    left_weight, right_weight, rel_tol=0.0, abs_tol=0.0
                ):
                    raise AnalysisError(
                        f"paired rows disagree on sampling weight: {dataset}/{left['input']['system_id']}"
                    )
                weighted_tables[(dataset, subset)][outcome] += left_weight
        for (dataset, subset), table in sorted(tables.items()):
            total = sum(table.values())
            weighted_table = weighted_tables.get((dataset, subset))
            weighted_total = (
                sum(weighted_table.values()) if weighted_table is not None else 0.0
            )
            paired_convergence[f"{label}/{dataset}/{subset}"] = {
                "comparison": label,
                "dataset": dataset,
                "subset": subset,
                "probability_estimate_eligible": subset != "stress",
                "paired_eligible": total,
                "unweighted_sample_table": {
                    name: table[name]
                    for name in ("both", "left_only", "right_only", "neither")
                },
                "unweighted_sample_paired_rate_difference": (
                    (table["left_only"] - table["right_only"]) / total
                    if total
                    else None
                ),
                "weighted_population_table": (
                    {
                        name: weighted_table[name]
                        for name in ("both", "left_only", "right_only", "neither")
                    }
                    if subset != "stress" and weighted_table is not None
                    else None
                ),
                "weighted_paired_rate_difference": (
                    (weighted_table["left_only"] - weighted_table["right_only"])
                    / weighted_total
                    if subset != "stress"
                    and weighted_table is not None
                    and weighted_total
                    else None
                ),
            }

    document = {
        "schema_version": 3,
        "gates": {**requested, "cpu_cuda": args.cpu_cuda_atol},
        "equivalence": equivalence,
        "equivalence_by_stratum": equivalence_by_stratum,
        "convergence": convergence,
        "convergence_by_stratum": convergence_by_stratum,
        "paired_convergence": paired_convergence,
        "outlier_count": len(outliers),
        "diagnostic_outlier_count": len(diagnostic_outliers),
        "completeness_failures": completeness_failures,
        "formal_gate": {
            "required": args.require_complete_p0,
            "status": "fail" if outliers or completeness_failures else "pass",
        },
    }
    for path in (args.output_json, args.outliers_csv, args.paired_errors_jsonl):
        if path.exists():
            raise AnalysisError(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with args.paired_errors_jsonl.open("x", encoding="utf-8", newline="\n") as handle:
        for row in paired_error_ledger:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
    with args.outliers_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "comparison",
                "dataset",
                "subset",
                "system_id",
                "quantity",
                "maximum",
                "reason",
                "severity",
            ),
        )
        writer.writeheader()
        writer.writerows([*outliers, *diagnostic_outliers])
    return 1 if outliers or completeness_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
