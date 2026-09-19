#!/usr/bin/env python3
"""SI persistent xTB/tblite N-process x 1-thread sensitivity matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

try:
    import resource
except (
    ImportError
):  # CLI/help remains inspectable on Windows; experiments are Linux-only.
    resource = None  # type: ignore[assignment]


class PoolError(RuntimeError):
    pass


def current_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def parse_ao_bin(value: str) -> tuple[int, int | None, str]:
    low_text, high_text = value.split("-", 1)
    low = int(low_text)
    high = None if high_text == "inf" else int(high_text)
    if low < 1 or (high is not None and high < low):
        raise argparse.ArgumentTypeError(f"invalid AO bin: {value}")
    return low, high, value


def ao_count(item: Any) -> int:
    value = item.system.manifest.get("gfn2_n_ao") or item.system.strata.get("gfn2_n_ao")
    if value in (None, ""):
        raise PoolError(f"{item.system_id} lacks frozen gfn2_n_ao")
    return int(value)


def worker(
    connection: Any,
    repo: str,
    systems: list[Any],
    engine: str,
    library: str,
    cpu: int,
    max_scc_iterations: int,
    accuracy: float,
    electronic_temperature_kelvin: float,
) -> None:
    try:
        if resource is None or not hasattr(os, "sched_setaffinity"):
            raise PoolError("persistent process-pool experiment requires Linux")
        os.sched_setaffinity(0, {cpu})
        from paper_runtime import install

        install(Path(repo))
        import dataset_runner  # type: ignore

        if engine == "xtb":
            from xtb_adapter import XtbAdapter as Adapter  # type: ignore

            adapter = Adapter(
                Path(library),
                dataset_runner.storage_from_systems(systems),
                "force",
                None,
                accuracy=accuracy,
                max_iterations=max_scc_iterations,
                electronic_temperature_kelvin=electronic_temperature_kelvin,
                threads=1,
            )
        else:
            from tblite_adapter import TbliteAdapter as Adapter  # type: ignore

            adapter = Adapter(
                Path(library),
                dataset_runner.storage_from_systems(systems),
                "force",
                accuracy=accuracy,
                max_iterations=max_scc_iterations,
                electronic_temperature_hartree=electronic_temperature_kelvin
                * 3.166808578545117e-6,
                collect_atomic_charges=False,
                threads=1,
            )
        connection.send(
            {
                "state": "ready",
                "pid": os.getpid(),
                "cpu": cpu,
                "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
        )
        while True:
            command = connection.recv()
            if command == "close":
                adapter.close()
                return
            before_cpu = time.process_time_ns()
            start = time.perf_counter_ns()
            adapter.restart_scc()
            adapter.invoke()
            result = adapter.results()
            elapsed = time.perf_counter_ns() - start
            connection.send(
                {
                    "state": "done",
                    "elapsed_ns": elapsed,
                    "cpu_ns": time.process_time_ns() - before_cpu,
                    "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "systems": len(systems),
                    "result": result,
                }
            )
    except BaseException as exc:  # noqa: BLE001 - worker must report every Python exit
        connection.send({"state": "error", "error": f"{type(exc).__name__}: {exc}"})


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = int(index), min(len(values) - 1, int(index) + 1)
    fraction = index - low
    return values[low] * (1 - fraction) + values[high] * fraction


def timing_summary(
    samples: list[dict[str, Any]], draws: int, seed: str, systems: int
) -> dict[str, Any]:
    wall = [row["wall_ms"] for row in samples]
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choice(wall) for _ in wall) for _ in range(draws)
    ]
    median_ms = statistics.median(wall)
    return {
        "median_wall_ms": median_ms,
        "iqr_wall_ms": percentile(wall, 0.75) - percentile(wall, 0.25),
        "bootstrap_95_ci_median_ms": [
            percentile(medians, 0.025),
            percentile(medians, 0.975),
        ],
        "median_systems_per_second": systems / (median_ms * 1e-3),
        "median_aggregate_cpu_ms": statistics.median(
            row["aggregate_cpu_ms"] for row in samples
        ),
        "sampled_peak_concurrent_rss_bytes": max(
            row["sampled_peak_concurrent_rss_bytes"] for row in samples
        ),
        "sum_worker_hwm_bytes_upper_bound": max(
            row["sum_worker_hwm_bytes_upper_bound"] for row in samples
        ),
        "rss_sampling_interval_seconds": 0.002,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("qm9", "omol25"), required=True)
    parser.add_argument("--subset", default="performance")
    parser.add_argument("--engine", choices=("xtb", "tblite"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--processes", type=parse_csv_ints, required=True)
    parser.add_argument("--ao-bin", type=parse_ao_bin, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--energy-atol", type=float, default=5e-7)
    parser.add_argument("--force-atol", type=float, default=5e-6)
    parser.add_argument("--charge-atol", type=float, default=5e-7)
    parser.add_argument("--max-scc-iterations", type=int, required=True)
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--electronic-temperature-kelvin", type=float, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise PoolError(f"refusing to overwrite: {args.output}")
    if args.max_scc_iterations <= 0:
        raise PoolError("max SCC iterations must be positive")
    if not math.isfinite(args.accuracy) or args.accuracy <= 0.0:
        raise PoolError("accuracy must be finite and positive")
    if (
        not math.isfinite(args.electronic_temperature_kelvin)
        or args.electronic_temperature_kelvin < 0.0
    ):
        raise PoolError("electronic temperature must be finite and nonnegative")
    if args.energy_atol > 5e-7 or args.force_atol > 5e-6 or args.charge_atol > 5e-7:
        raise PoolError("requested correctness gate widens the paper contract")
    if not args.library.is_file():
        raise PoolError(f"library is missing: {args.library}")
    from paper_runtime import install, physical_cpu_ids

    install(args.repo)
    import dataset_runner  # type: ignore

    low, high, ao_label = args.ao_bin
    items = [
        item
        for item in dataset_runner.load_manifest(args.manifest, args.dataset)
        if item.system is not None
        and item.subset == args.subset
        and ao_count(item) >= low
        and (high is None or ao_count(item) <= high)
    ]
    # Match performance_matrix.select_systems exactly so the external pools and
    # xTBloom worker-pool rows use identical real structures.
    items.sort(
        key=lambda item: hashlib.sha256(
            f"{args.seed}\0{item.system_id}".encode()
        ).hexdigest()
    )
    if len(items) < args.batch_size:
        status = "not-applicable" if not items else "sample-censored"
        document = {
            "schema_version": 2,
            "max_scc_iterations": args.max_scc_iterations,
            "accuracy": args.accuracy,
            "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
            "claim_eligible": False,
            "selection_status": status,
            "formal_gate": {
                "status": "pass",
                "scope": "evidence-integrity only; no process-pool performance claim",
                "fatal_row_indices": [],
            },
            "rows": [
                {
                    "engine": args.engine,
                    "dataset": args.dataset,
                    "ao_bin": ao_label,
                    "batch_size": args.batch_size,
                    "available_distinct_systems": len(items),
                    "availability": status,
                    "claim_eligible": False,
                    "reason": f"need {args.batch_size} distinct systems, found {len(items)}",
                }
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return 0
    references: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(args.reference_root.rglob("results.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (
                    row["program"]["engine"] == args.engine
                    and row["status"]["category"] == "success"
                ):
                    references[(row["input"]["dataset"], row["input"]["system_id"])] = (
                        row
                    )
    selected_items = items[: args.batch_size]
    missing = [
        item.system_id
        for item in selected_items
        if (args.dataset, item.system_id) not in references
    ]
    if missing:
        raise PoolError(
            f"selected pool coordinate lacks successful frozen references: {missing[:8]}"
        )
    systems = [item.system for item in selected_items]
    allowed_cpus = physical_cpu_ids()
    context = mp.get_context("spawn")
    rows = []
    for count in args.processes:
        if count > len(allowed_cpus) or count > len(systems):
            rows.append(
                {
                    "processes": count,
                    "availability": "unavailable",
                    "reason": "allocation lacks CPUs or systems",
                }
            )
            continue
        chunks = [systems[index::count] for index in range(count)]
        chunk_items = [selected_items[index::count] for index in range(count)]
        parents = []
        processes = []
        startup = time.perf_counter_ns()
        for index, chunk in enumerate(chunks):
            parent, child = context.Pipe()
            process = context.Process(
                target=worker,
                args=(
                    child,
                    str(args.repo),
                    chunk,
                    args.engine,
                    str(args.library),
                    allowed_cpus[index],
                    args.max_scc_iterations,
                    args.accuracy,
                    args.electronic_temperature_kelvin,
                ),
            )
            process.daemon = True
            process.start()
            child.close()
            parents.append(parent)
            processes.append(process)
        ready = [connection.recv() for connection in parents]
        startup_ms = (time.perf_counter_ns() - startup) * 1e-6
        if any(message["state"] != "ready" for message in ready):
            raise PoolError(f"worker setup failed: {ready}")
        worker_pids = [int(message["pid"]) for message in ready]

        def invoke(
            parent_connections: tuple[Any, ...] = tuple(parents),
            item_chunks: tuple[tuple[Any, ...], ...] = tuple(
                tuple(chunk) for chunk in chunk_items
            ),
            worker_process_ids: tuple[int, ...] = tuple(worker_pids),
        ) -> dict[str, Any]:
            parent_cpu_start = time.process_time_ns()
            start = time.perf_counter_ns()
            for connection in parent_connections:
                connection.send("run")
            responses: list[dict[str, Any] | None] = [None] * len(parent_connections)
            pending = set(range(len(parent_connections)))
            rss_samples: list[dict[str, int]] = []
            rss_sample_count = 0
            while pending:
                # The deployable reference is the complete process group, not
                # just its workers: include this coordinating parent in the
                # concurrent RSS sample taken on the same time axis.
                aggregate_rss = current_rss_bytes(os.getpid()) + sum(
                    current_rss_bytes(pid) for pid in worker_process_ids
                )
                rss_sample_count += 1
                row = {
                    "elapsed_ns": time.perf_counter_ns() - start,
                    "aggregate_current_rss_bytes": aggregate_rss,
                }
                if (
                    not rss_samples
                    or aggregate_rss != rss_samples[-1]["aggregate_current_rss_bytes"]
                    or rss_sample_count % 100 == 0
                ):
                    rss_samples.append(row)
                completed = []
                for index in pending:
                    if parent_connections[index].poll():
                        responses[index] = parent_connections[index].recv()
                        completed.append(index)
                pending.difference_update(completed)
                if pending:
                    time.sleep(0.002)
            wall_ns = time.perf_counter_ns() - start
            parent_cpu_ns = time.process_time_ns() - parent_cpu_start
            assert all(response is not None for response in responses)
            complete_responses = [
                response for response in responses if response is not None
            ]
            if any(message["state"] != "done" for message in complete_responses):
                raise PoolError(f"worker inference failed: {complete_responses}")
            peak_concurrent_rss = max(
                (sample["aggregate_current_rss_bytes"] for sample in rss_samples),
                default=0,
            )
            if peak_concurrent_rss <= 0:
                raise PoolError(
                    "concurrent worker RSS sampling produced no usable value"
                )
            maximums = {"energy": 0.0, "forces": 0.0}
            result_names = {
                "energy": "energies_hartree",
                "forces": "forces_hartree_per_bohr",
            }
            for members, response in zip(item_chunks, complete_responses, strict=True):
                result = response["result"]
                atom_cursor = 0
                for index, item in enumerate(members):
                    reference = references[(args.dataset, item.system_id)]["results"]
                    atoms = len(item.system.atomic_numbers)
                    for quantity, result_name in result_names.items():
                        field = reference.get(quantity, {})
                        if (
                            field.get("availability") != "available"
                            or result_name not in result
                        ):
                            raise PoolError(
                                f"missing required {quantity} for {item.system_id}"
                            )
                        expected = field["value"]
                        if quantity == "energy":
                            actual_values = [float(result[result_name][index])]
                            expected_values = [float(expected)]
                        else:
                            flat_expected: list[float] = []
                            stack = (
                                list(expected)
                                if isinstance(expected, list)
                                else [expected]
                            )
                            while stack:
                                value = stack.pop(0)
                                if isinstance(value, list):
                                    stack[0:0] = value
                                else:
                                    flat_expected.append(float(value))
                            width = (3 if quantity == "forces" else 1) * atoms
                            begin = (3 if quantity == "forces" else 1) * atom_cursor
                            actual_values = [
                                float(value)
                                for value in result[result_name][begin : begin + width]
                            ]
                            expected_values = flat_expected
                        if (
                            len(actual_values) != len(expected_values)
                            or not actual_values
                        ):
                            raise PoolError(
                                f"shape mismatch for {quantity}: {item.system_id}"
                            )
                        if not all(
                            math.isfinite(value)
                            for value in (*actual_values, *expected_values)
                        ):
                            raise PoolError(f"non-finite {quantity}: {item.system_id}")
                        maximums[quantity] = max(
                            maximums[quantity],
                            max(
                                abs(a - b)
                                for a, b in zip(
                                    actual_values, expected_values, strict=True
                                )
                            ),
                        )
                    atom_cursor += atoms
            limits = {"energy": args.energy_atol, "forces": args.force_atol}
            if any(maximums[name] > limits[name] for name in limits):
                raise PoolError(f"process-pool correctness gate failed: {maximums}")
            return {
                "wall_ms": wall_ns * 1e-6,
                "aggregate_cpu_ms": (
                    parent_cpu_ns
                    + sum(message["cpu_ns"] for message in complete_responses)
                )
                * 1e-6,
                "parent_cpu_ms": parent_cpu_ns * 1e-6,
                "sampled_peak_concurrent_rss_bytes": peak_concurrent_rss,
                "sum_worker_hwm_bytes_upper_bound": sum(
                    message["rss"] for message in complete_responses
                ),
                "rss_sample_count": rss_sample_count,
                "retained_rss_samples": rss_samples,
                "correctness_max_abs": maximums,
            }

        for _ in range(args.warmups):
            invoke()
        samples = [invoke() for _ in range(args.repetitions)]
        for connection in parents:
            connection.send("close")
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise PoolError(f"worker exited with {process.exitcode}")
        rows.append(
            {
                "engine": args.engine,
                "execution": "persistent-process-pool",
                "dataset": args.dataset,
                "ao_bin": ao_label,
                "system_ids": [item.system_id for item in items[: args.batch_size]],
                "ao_counts": [ao_count(item) for item in items[: args.batch_size]],
                "atom_counts": [
                    len(item.system.atomic_numbers) for item in items[: args.batch_size]
                ],
                "processes": count,
                "threads_per_process": 1,
                "cpus": allowed_cpus[:count],
                "batch_size": args.batch_size,
                "max_scc_iterations": args.max_scc_iterations,
                "accuracy": args.accuracy,
                "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
                "timed_output_contract": "energy+analytic-forces",
                "startup_ms": startup_ms,
                "raw_samples": samples,
                "summary": timing_summary(
                    samples,
                    args.bootstrap_samples,
                    f"{args.seed}|{args.engine}|{count}",
                    args.batch_size,
                ),
                "availability": "available",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fatal_rows = [
        index
        for index, row in enumerate(rows)
        if row.get("availability") != "available"
    ]
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "max_scc_iterations": args.max_scc_iterations,
                "accuracy": args.accuracy,
                "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
                "claim_eligible": bool(rows) and not fatal_rows,
                "formal_gate": {
                    "status": "fail" if fatal_rows else "pass",
                    "fatal_row_indices": fatal_rows,
                },
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 1 if fatal_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
