#!/usr/bin/env python3
"""Render the gpuxtb cross-engine scaling figures from benchmark artifacts.

The script merges every ``--output-json`` artifact produced by
``natoms_cross_engine.py`` (CPU and CUDA runs, one file per engine set),
keeps only correctness-qualified ``available`` rows, and draws:

1. ``batch=1``: steady-state GFN2-xTB energy+force latency vs molecule size
   (all engines that succeeded; gpuxtb lines highlighted);
2. ``batch=128``: same latency axis for ragged batches of 128 *distinct*
   systems (gpuxtb highlighted);
3. MD-trajectory: per-frame latency vs molecule size for nearly identical
   coordinate sequences (gpuxtb WARM highlighted).

Every panel records the exact hardware (CPU model + worker count, GPU model)
and the repository commit.  Reference engines without a matching point are
omitted from that panel; no invented numbers are inserted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _engine_label(engine: str) -> str:
    labels = {
        "gpuxtb-cpu": "gpuxtb (CPU)",
        "gpuxtb-cuda": "gpuxtb (CUDA)",
        "xtb": "xTB",
        "tblite": "tblite",
        "dxtb-cpu": "dxtb (CPU)",
        "dxtb-cuda": "dxtb (CUDA)",
    }
    return labels.get(engine, engine)


def _engine_color(engine: str) -> str:
    colors = {
        "gpuxtb-cpu": "#d62728",
        "gpuxtb-cuda": "#ff7f0e",
        "xtb": "#7f7f7f",
        "tblite": "#bcbd22",
        "dxtb-cpu": "#7b7b7b",
        "dxtb-cuda": "#9e9e9e",
    }
    return colors.get(engine, "#555555")


def load_rows(
    artifact_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge all artifact rows and the first complete metadata block."""
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for path in artifact_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(document.get("rows", []))
        if not metadata and "metadata" in document:
            metadata = document["metadata"]
    return rows, metadata


def _median_ms(row: dict[str, Any]) -> float | None:
    timing = row.get("timing") or {}
    value = timing.get("median_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _is_eligible(row: dict[str, Any]) -> bool:
    """Keep only successful, correctness-qualified, finite rows."""
    if row.get("availability") != "available":
        return False
    median = _median_ms(row)
    if median is None or median <= 0.0 or not (median == median):
        return False
    correctness = row.get("correctness")
    if correctness and correctness.get("status") != "pass":
        return False
    return True


def _annotate_hardware(axes: Any, metadata: dict[str, Any], commit: str) -> None:
    """Draw the hardware + threads + commit box inside the axes."""
    hardware = metadata.get("hardware", {})
    threads = metadata.get("threads", {})
    gpu = hardware.get("nvidia_smi")
    lines = [f"commit {commit[:12]}"]
    cpu_model = hardware.get("cpu_model")
    if cpu_model:
        lines.append(f"{cpu_model}")
        lines.append(f"CPU threads: {threads.get('cpu_threads', '?')}")
    if gpu:
        lines.append(gpu.replace("\n", " "))
    lines.append("gpuxtb SCC 1e-10/1e-12; refs acc 1e-4")
    props = dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="#888888",
        alpha=0.95,
    )
    axes.text(
        0.98,
        0.03,
        "\n".join(lines),
        transform=axes.transAxes,
        fontsize=7,
        va="bottom",
        ha="right",
        family="monospace",
        bbox=props,
    )


def _scaling_panel(
    axes: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    engines: list[str],
    x_scale: str,
) -> None:
    """Draw one batch-size scaling panel with gpuxtb highlighted."""
    for engine in engines:
        qualified = [
            row
            for row in rows
            if _is_eligible(row)
            and row.get("engine") == engine
            and row.get("batch_size") == batch_size
            and "job" not in row
        ]
        qualified.sort(key=lambda row: row["natoms"])
        if not qualified:
            continue
        x_values = [float(row["natoms"]) for row in qualified]
        y_values = [_median_ms(row) for row in qualified if _median_ms(row) is not None]
        if len(x_values) != len(y_values):
            continue
        highlight = engine.startswith("gpuxtb")
        axes.loglog(
            x_values,
            y_values,
            marker="o",
            markersize=(9 if highlight else 5),
            linewidth=(3.2 if highlight else 1.6),
            linestyle="-",
            color=_engine_color(engine),
            zorder=(5 if highlight else 2),
            label=_engine_label(engine),
        )
        if not highlight:
            axes.scatter(
                x_values,
                y_values,
                marker="x",
                s=14,
                color=_engine_color(engine),
                zorder=3,
            )
    axes.set_xlabel("molecule size (atoms)")
    axes.set_ylabel("energy + force latency (ms)")
    if batch_size == 128:
        axes.set_title(f"batch size = {batch_size} (128 distinct systems per call)")
    else:
        axes.set_title(f"batch size = {batch_size}")
    axes.grid(True, which="both", ls=":", alpha=0.5)
    axes.set_xscale(x_scale)


def _trajectory_panel(
    axes: Any,
    rows: list[dict[str, Any]],
    engines: list[str],
) -> None:
    """Draw per-frame MD-trajectory latency vs molecule size, gpuxtb WARM."""
    for engine in engines:
        qualified = [
            row
            for row in rows
            if _is_eligible(row)
            and row.get("engine") == engine
            and row.get("job") == "trajectory"
            and row.get("batch_size") == 1
        ]
        qualified.sort(key=lambda row: row["natoms"])
        if not qualified:
            continue
        x_values = [float(row["natoms"]) for row in qualified]
        y_values = [_median_ms(row) for row in qualified if _median_ms(row) is not None]
        if len(x_values) != len(y_values) or len(x_values) < 1:
            continue
        highlight = engine.startswith("gpuxtb")
        axes.loglog(
            x_values,
            y_values,
            marker="o",
            markersize=(9 if highlight else 5),
            linewidth=(3.2 if highlight else 1.6),
            linestyle="-",
            color=_engine_color(engine),
            zorder=(5 if highlight else 2),
            label=_engine_label(engine),
        )
        if not highlight:
            axes.scatter(
                x_values,
                y_values,
                marker="x",
                s=14,
                color=_engine_color(engine),
                zorder=3,
            )
    axes.set_xlabel("molecule size (atoms)")
    axes.set_ylabel("per-frame latency (ms)")
    axes.set_title(
        "MD-style trajectory (nearly identical frames), gpuxtb = WARM SCC continuation"
    )
    axes.grid(True, which="both", ls=":", alpha=0.5)
    axes.set_xscale("log")
    axes.set_yscale("log")


def build_parser() -> argparse.ArgumentParser:
    """Configure the figure CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engines",
        nargs="*",
        default=[
            "gpuxtb-cpu",
            "gpuxtb-cuda",
            "xtb",
            "tblite",
            "dxtb-cpu",
            "dxtb-cuda",
        ],
    )
    parser.add_argument("--commit", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Merge artifacts and write the three-panel figure."""
    args = build_parser().parse_args(argv)
    if not args.artifact:
        print("ERROR: at least one --artifact JSON is required", file=sys.stderr)
        return 2
    rows, metadata = load_rows(args.artifact)
    repo_root = Path(__file__).resolve().parents[1]
    commit = args.commit
    if commit is None:
        without_dot_git = metadata.get("commit", {}) or {}
        commit = without_dot_git.get("head") or "unknown"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"ERROR: matplotlib is required for plotting: {exc}", file=sys.stderr)
        return 2

    fig, axes_list = plt.subplots(
        1, 3, figsize=(18, 6), squeeze=False, constrained_layout=True
    )
    _scaling_panel(axes_list[0, 0], rows, 1, args.engines, "log")
    _scaling_panel(axes_list[0, 1], rows, 128, args.engines, "log")
    _trajectory_panel(axes_list[0, 2], rows, args.engines)
    for axes in axes_list[0]:
        _annotate_hardware(axes, metadata, commit)

    handles, labels = axes_list[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            frameon=True,
            bbox_to_anchor=(0.5, 1.06),
        )
    title = (
        "gpuxtb GFN2-xTB energy + force inference scaling\n"
        "batch = 1 (single molecule) | batch = 128 (128 distinct systems per batch) | trajectory (WARM)"
    )
    fig.suptitle(title, fontsize=13, y=1.18 if handles else 1.02)
    if args.output.suffix == ".svg":
        fig.savefig(args.output, format="svg", bbox_inches="tight")
    else:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
