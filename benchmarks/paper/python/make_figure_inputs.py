#!/usr/bin/env python3
"""Build compact Table 1 and Figure 2-4 CSV inputs from raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def performance_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in document.get("rows", []):
            timing = row.get("steady_fresh", {})
            worker = row.get("worker_scaling", {})
            scc = row.get("scc_iterations", {})
            memory = row.get("memory", {})
            rows.append(
                {
                    "source": str(path.relative_to(root)),
                    "suite": row.get("suite"),
                    "hardware_id": row.get("hardware_id"),
                    "dataset": row.get("dataset"),
                    "engine": row.get("engine"),
                    "backend": row.get("backend"),
                    "memory_mode": row.get("memory_mode"),
                    "ao_bin": row.get("ao_bin"),
                    "batch_size": row.get("batch_size"),
                    "cpu_threads": row.get("cpu_threads"),
                    "availability": row.get("availability"),
                    "median_ms": timing.get("median_ms"),
                    "iqr_ms": timing.get("iqr_ms"),
                    "p95_ms": timing.get("p95_ms"),
                    "median_ci_low_ms": (
                        timing.get("bootstrap_95_ci_median_ms") or [None, None]
                    )[0],
                    "median_ci_high_ms": (
                        timing.get("bootstrap_95_ci_median_ms") or [None, None]
                    )[1],
                    "median_process_cpu_ms": timing.get("median_process_cpu_ms"),
                    "median_cpu_utilization_cores": timing.get(
                        "median_cpu_utilization_cores"
                    ),
                    "systems_per_second": timing.get("median_systems_per_second"),
                    "successful_systems_per_second": timing.get(
                        "successful_systems_per_second"
                    ),
                    "matched_success_systems_per_second": timing.get(
                        "matched_success_systems_per_second"
                    ),
                    "success_rate": timing.get("success_rate"),
                    "correctness": row.get("correctness", {}).get("status"),
                    "worker_speedup_vs_one_thread": worker.get("speedup_vs_one_thread"),
                    "parallel_efficiency": worker.get("parallel_efficiency"),
                    "scc_iteration_median": scc.get("median"),
                    "scc_iteration_p95": scc.get("p95"),
                    "memory_probe_status": memory.get("status"),
                    "sampled_peak_current_rss_bytes": memory.get(
                        "sampled_peak_current_rss_bytes"
                    ),
                    "sampled_incremental_rss_bytes": memory.get(
                        "sampled_incremental_rss_bytes"
                    ),
                    "sampled_peak_process_gpu_memory_bytes": memory.get(
                        "sampled_peak_process_gpu_memory_bytes"
                    ),
                    "memory_sampling_interval_seconds": memory.get(
                        "sampling_interval_seconds"
                    ),
                    "atom_count_min": min(row.get("atom_counts", []), default=None),
                    "atom_count_max": max(row.get("atom_counts", []), default=None),
                    "ao_count_min": min(row.get("ao_counts", []), default=None),
                    "ao_count_max": max(row.get("ao_counts", []), default=None),
                    "pair_count_25_bohr": row.get("geometric_pair_counts", {})
                    .get("batch_total_by_cutoff_bohr", {})
                    .get("25"),
                    "pair_count_30_bohr": row.get("geometric_pair_counts", {})
                    .get("batch_total_by_cutoff_bohr", {})
                    .get("30"),
                    "pair_count_50_bohr": row.get("geometric_pair_counts", {})
                    .get("batch_total_by_cutoff_bohr", {})
                    .get("50"),
                    "_raw_ms": timing.get("raw_ms", []),
                }
            )
    lookup = {
        (
            row["source"],
            row["dataset"],
            row["ao_bin"],
            row["batch_size"],
            row["cpu_threads"],
            row["engine"],
            row["backend"],
            row["memory_mode"],
        ): row
        for row in rows
        if row["availability"] == "available"
    }

    def annotate(
        candidate: dict[str, Any], label: str, baseline_key: tuple[Any, ...]
    ) -> None:
        baseline = lookup.get(baseline_key)
        if baseline is None or not candidate["_raw_ms"] or not baseline["_raw_ms"]:
            return
        seed = hashlib.sha256(
            f"{candidate['source']}|{label}|{candidate['engine']}|{candidate['ao_bin']}|{candidate['batch_size']}|{candidate['cpu_threads']}".encode()
        ).hexdigest()
        generator = random.Random(seed)
        ratios = []
        for _ in range(10_000):
            base_median = statistics.median(
                generator.choice(baseline["_raw_ms"]) for _ in baseline["_raw_ms"]
            )
            candidate_median = statistics.median(
                generator.choice(candidate["_raw_ms"]) for _ in candidate["_raw_ms"]
            )
            ratios.append(base_median / candidate_median)
        ratios.sort()
        candidate[label] = statistics.median(baseline["_raw_ms"]) / statistics.median(
            candidate["_raw_ms"]
        )
        candidate[f"{label}_ci_low"] = ratios[int(0.025 * (len(ratios) - 1))]
        candidate[f"{label}_ci_high"] = ratios[int(0.975 * (len(ratios) - 1))]

    for row in rows:
        prefix = (
            row["source"],
            row["dataset"],
            row["ao_bin"],
            row["batch_size"],
            row["cpu_threads"],
        )
        if row["engine"] == "xtbloom-ragged" and row["backend"] == "cpu":
            annotate(row, "speedup_vs_xtb", (*prefix, "xtb", "cpu", "host"))
            annotate(row, "speedup_vs_tblite", (*prefix, "tblite", "cpu", "host"))
            annotate(
                row,
                "speedup_vs_sequential",
                (*prefix, "xtbloom-sequential", "cpu", "host"),
            )
        if row["engine"] == "xtbloom-ragged" and row["backend"] == "cuda":
            annotate(row, "speedup_vs_cpu", (*prefix, "xtbloom-ragged", "cpu", "host"))
    for row in rows:
        row.pop("_raw_ms", None)
    return rows


def bootstrap_speedup(
    baseline_samples: list[float], candidate_samples: list[float], seed: str
) -> tuple[float, float, float]:
    generator = random.Random(hashlib.sha256(seed.encode()).hexdigest())
    draws = []
    for _ in range(10_000):
        baseline = statistics.median(
            generator.choice(baseline_samples) for _ in baseline_samples
        )
        candidate = statistics.median(
            generator.choice(candidate_samples) for _ in candidate_samples
        )
        draws.append(baseline / candidate)
    draws.sort()
    return (
        statistics.median(baseline_samples) / statistics.median(candidate_samples),
        draws[int(0.025 * (len(draws) - 1))],
        draws[int(0.975 * (len(draws) - 1))],
    )


def process_pool_comparisons(root: Path) -> list[dict[str, Any]]:
    stage = root / "si-cpu-process-pool" / "raw"
    baselines: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for path in sorted(stage.glob("*-xtbloom.json")):
        document = json.loads(path.read_text())
        for row in document.get("rows", []):
            timing = row.get("steady_fresh", {})
            if row.get("availability") != "available" or not timing.get("raw_ms"):
                continue
            key = (
                row["dataset"],
                row["ao_bin"],
                int(row["batch_size"]),
                int(row["cpu_threads"]),
            )
            baselines[key] = row
    output = []
    for path in sorted(stage.glob("*.json")):
        if path.name.endswith("-xtbloom.json"):
            continue
        document = json.loads(path.read_text())
        for row in document.get("rows", []):
            summary = row.get("summary", {})
            samples = [
                float(sample["wall_ms"]) for sample in row.get("raw_samples", [])
            ]
            if row.get("availability") != "available" or not samples:
                continue
            key = (
                row["dataset"],
                row["ao_bin"],
                int(row["batch_size"]),
                int(row["processes"]),
            )
            baseline = baselines.get(key)
            if baseline is None:
                continue
            baseline_samples = [
                float(value) for value in baseline["steady_fresh"]["raw_ms"]
            ]
            speedup, low, high = bootstrap_speedup(
                samples,
                baseline_samples,
                f"pool|{row['dataset']}|{row['ao_bin']}|{row['batch_size']}|{row['processes']}|{row['engine']}",
            )
            output.append(
                {
                    "source": str(path.relative_to(root)),
                    "dataset": row["dataset"],
                    "ao_bin": row["ao_bin"],
                    "batch_size": row["batch_size"],
                    "core_budget": row["processes"],
                    "reference_engine": row["engine"],
                    "reference_median_wall_ms": summary.get("median_wall_ms"),
                    "reference_median_systems_per_second": summary.get(
                        "median_systems_per_second"
                    ),
                    "reference_median_aggregate_cpu_ms": summary.get(
                        "median_aggregate_cpu_ms"
                    ),
                    "reference_sampled_peak_concurrent_rss_bytes": summary.get(
                        "sampled_peak_concurrent_rss_bytes"
                    ),
                    "reference_sum_worker_hwm_bytes_upper_bound": summary.get(
                        "sum_worker_hwm_bytes_upper_bound"
                    ),
                    "reference_startup_ms": row.get("startup_ms"),
                    "xtbloom_median_ms": baseline["steady_fresh"].get("median_ms"),
                    "xtbloom_median_systems_per_second": baseline["steady_fresh"].get(
                        "median_systems_per_second"
                    ),
                    "xtbloom_peak_current_rss_bytes": baseline.get("memory", {}).get(
                        "sampled_peak_current_rss_bytes"
                    ),
                    "xtbloom_speedup": speedup,
                    "xtbloom_speedup_bootstrap_ci_low": low,
                    "xtbloom_speedup_bootstrap_ci_high": high,
                    "xtbloom_faster_claim_eligible": low > 1.0,
                }
            )
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in output:
        key = (
            row["dataset"],
            row["ao_bin"],
            int(row["batch_size"]),
            int(row["core_budget"]),
        )
        groups.setdefault(key, []).append(row)
    for members in groups.values():
        best = min(
            members,
            key=lambda row: float(row["reference_median_wall_ms"]),
        )
        for row in members:
            row["best_external_reference_at_core_budget"] = row is best
    return output


def finite_difference_rows(root: Path) -> list[dict[str, Any]]:
    """Normalize every preregistered finite-difference coordinate for the SI."""
    output: list[dict[str, Any]] = []
    for stage in ("p0c-fd-cpu", "p0c-fd-gpu"):
        for path in sorted((root / stage / "derived").glob("*.json")):
            document = json.loads(path.read_text())
            gate = document.get("atol_hartree_per_bohr")
            for row in document.get("rows", []):
                output.append(
                    {
                        "source": str(path.relative_to(root)),
                        "gate_hartree_per_bohr": gate,
                        **row,
                    }
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis_path = (
        args.run_root / "exp3a-convergence" / "derived" / "table1-convergence.json"
    )
    if not analysis_path.is_file():
        raise RuntimeError(f"missing convergence analysis: {analysis_path}")
    analysis = json.loads(analysis_path.read_text())
    table1 = []
    for comparison, document in analysis["equivalence"].items():
        for quantity, values in document["quantities"].items():
            table1.append(
                {
                    "panel": "A",
                    "comparison": comparison,
                    "dataset": document.get("dataset"),
                    "subset": document.get("subset"),
                    "probability_estimate_eligible": document.get(
                        "probability_estimate_eligible"
                    ),
                    "quantity": quantity,
                    **values,
                }
            )
    for coordinate, values in analysis["convergence"].items():
        if values.get("subset") not in {"main", "stress"}:
            continue
        scc = values.get("scc_iterations", {})
        wilson = values.get("wilson_95_unweighted_sample_rate") or [None, None]
        weighted_wilson = values.get("weighted_kish_wilson_95_approx") or [None, None]
        table1.append(
            {
                "panel": "B",
                "coordinate": coordinate,
                **{
                    key: value
                    for key, value in values.items()
                    if key
                    not in {
                        "categories",
                        "scc_iterations",
                        "wilson_95_unweighted_sample_rate",
                        "weighted_kish_wilson_95_approx",
                    }
                },
                "scc_iteration_median": scc.get("median"),
                "scc_iteration_p95": scc.get("p95"),
                "wilson_95_unweighted_sample_rate_low": wilson[0],
                "wilson_95_unweighted_sample_rate_high": wilson[1],
                "weighted_kish_wilson_95_approx_low": weighted_wilson[0],
                "weighted_kish_wilson_95_approx_high": weighted_wilson[1],
                "categories_json": json.dumps(
                    values.get("categories", {}), sort_keys=True
                ),
            }
        )
    write_csv(args.output_dir / "table1.csv", table1)

    paired_rows = []
    for coordinate, values in analysis.get("paired_convergence", {}).items():
        unweighted = values.get("unweighted_sample_table") or {}
        weighted = values.get("weighted_population_table") or {}
        paired_rows.append(
            {
                "coordinate": coordinate,
                **{
                    key: value
                    for key, value in values.items()
                    if key
                    not in {"unweighted_sample_table", "weighted_population_table"}
                },
                **{f"unweighted_{key}": value for key, value in unweighted.items()},
                **{f"weighted_{key}": value for key, value in weighted.items()},
            }
        )
    write_csv(args.output_dir / "figure4-paired-convergence.csv", paired_rows)

    scc_ecdf_rows = []
    for coordinate, values in analysis["convergence"].items():
        for point in values.get("scc_iterations", {}).get("ecdf", []):
            scc_ecdf_rows.append(
                {
                    "coordinate": coordinate,
                    "dataset": values.get("dataset"),
                    "subset": values.get("subset"),
                    "engine": values.get("engine"),
                    "execution": values.get("execution"),
                    **point,
                }
            )
    write_csv(args.output_dir / "figure4-scc-ecdf.csv", scc_ecdf_rows)

    stratum_rows = []
    for coordinate, values in analysis.get("convergence_by_stratum", {}).items():
        scc = values.get("scc_iterations", {})
        stratum_rows.append(
            {
                "coordinate": coordinate,
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"categories", "scc_iterations"}
                },
                "categories_json": json.dumps(
                    values.get("categories", {}), sort_keys=True
                ),
                "scc_iteration_median": scc.get("median"),
                "scc_iteration_p95": scc.get("p95"),
            }
        )
    write_csv(args.output_dir / "si-convergence-by-stratum.csv", stratum_rows)

    equivalence_stratum_rows = []
    for coordinate, values in analysis.get("equivalence_by_stratum", {}).items():
        for quantity, distribution in values.get("quantities", {}).items():
            equivalence_stratum_rows.append(
                {
                    "coordinate": coordinate,
                    **{
                        key: value
                        for key, value in values.items()
                        if key != "quantities"
                    },
                    "quantity": quantity,
                    **distribution,
                }
            )
    write_csv(
        args.output_dir / "si-equivalence-by-stratum.csv", equivalence_stratum_rows
    )
    write_csv(
        args.output_dir / "si-finite-difference.csv",
        finite_difference_rows(args.run_root),
    )

    performance = performance_rows(args.run_root)
    write_csv(
        args.output_dir / "figure2-cpu-native.csv",
        (row for row in performance if row["suite"] == "cpu-native"),
    )
    write_csv(
        args.output_dir / "si-cpu-process-pool.csv",
        process_pool_comparisons(args.run_root),
    )
    figure3 = [
        row for row in performance if row["suite"] in {"gpu-crossover", "capacity"}
    ]
    write_csv(
        args.output_dir / "si-second-hardware.csv",
        (row for row in performance if row["suite"] == "second-hardware"),
    )
    cpu = {
        (row["dataset"], row["ao_bin"], row["batch_size"]): row["systems_per_second"]
        for row in figure3
        if row["backend"] == "cpu" and row["availability"] == "available"
    }
    for row in figure3:
        baseline = cpu.get((row["dataset"], row["ao_bin"], row["batch_size"]))
        row["gpu_cpu_throughput_ratio"] = (
            row["systems_per_second"] / baseline
            if row["backend"] == "cuda"
            and baseline
            and row["systems_per_second"] is not None
            else None
        )
    write_csv(args.output_dir / "figure3-gpu-crossover.csv", figure3)
    convergence_rows = [
        {
            "coordinate": coordinate,
            **{
                key: value
                for key, value in values.items()
                if key not in {"categories", "scc_iterations"}
            },
            "categories_json": json.dumps(values.get("categories", {}), sort_keys=True),
            "scc_iteration_median": values.get("scc_iterations", {}).get("median"),
            "scc_iteration_p95": values.get("scc_iterations", {}).get("p95"),
        }
        for coordinate, values in analysis["convergence"].items()
        if values.get("subset") in {"main", "stress"}
    ]
    si_performance_convergence = [
        {
            "coordinate": coordinate,
            **{
                key: value
                for key, value in values.items()
                if key not in {"categories", "scc_iterations"}
            },
            "categories_json": json.dumps(values.get("categories", {}), sort_keys=True),
            "scc_iteration_median": values.get("scc_iterations", {}).get("median"),
            "scc_iteration_p95": values.get("scc_iterations", {}).get("p95"),
        }
        for coordinate, values in analysis["convergence"].items()
        if values.get("subset") == "performance"
    ]
    write_csv(args.output_dir / "figure4-convergence.csv", convergence_rows)
    write_csv(
        args.output_dir / "si-performance-convergence.csv", si_performance_convergence
    )
    write_csv(
        args.output_dir / "figure4-ragged.csv",
        (row for row in performance if row["suite"] == "ragged"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
