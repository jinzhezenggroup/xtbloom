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

A footnote below the panels records the exact hardware (CPU model + worker
count, GPU model) and the repository commit.  Reference engines without a
matching point are omitted from that panel; no invented numbers are inserted.
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
        "xtb": "#1f77b4",
        "tblite": "#2ca02c",
        "dxtb-cpu": "#9467bd",
        "dxtb-cuda": "#17becf",
    }
    return colors.get(engine, "#555555")


def load_rows(
    artifact_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge all artifact rows and the first complete metadata block.

    Every row is annotated with ``_artifact_start_policy`` taken from its
    source artifact's ``metadata.protocol.start_policy`` so the batch=1
    cold-start panel can reject job-less rows that were measured under an
    auto-warm start policy (for example the steady-state sample row that a
    trajectory invocation leaks into its output file).
    """
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for path in artifact_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        protocol = (document.get("metadata") or {}).get("protocol") or {}
        start_policy = protocol.get("start_policy")
        for row in document.get("rows", []):
            row["_artifact_start_policy"] = start_policy
            rows.append(row)
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
    if median is None or median <= 0.0 or median != median:
        return False
    correctness = row.get("correctness")
    return not (correctness and correctness.get("status") != "pass")


def _footnote_lines(metadata: dict[str, Any], commit: str) -> list[str]:
    """Build the shared hardware/provenance footnote drawn once below panels.

    Keeping the common context (commit, hardware, threads, tolerances) in a
    single bottom footnote instead of a per-panel box keeps every panel clean
    and the figure uncluttered.
    """
    hardware = metadata.get("hardware", {})
    threads = metadata.get("threads", {})
    line: list[str] = [f"commit {commit[:12]}"]
    cpu_model = hardware.get("cpu_model")
    if cpu_model:
        line.append(cpu_model)
        line.append(f"CPU threads: {threads.get('cpu_threads', '?')} per engine")
    gpu = hardware.get("nvidia_smi")
    if gpu:
        line.append(gpu.replace("\n", " "))
    line.append("gpuxtb SCC 1e-10/1e-12; refs acc 1e-4")
    line.append("build: -O3 generic x86-64 (no -march=native); dxtb: PyTorch AVX2")
    line.append("gpuxtb batch=1: 1 of 16 workers active (outer-batch pool idle)")
    return line


def _cold_batch1_row(row: dict[str, Any]) -> bool:
    """Decide whether a batch=1 row belongs in the cold-start panel.

    Accept only job-less rows that were not measured under an auto-warm start
    policy.

    Trajectory-mode invocations also run a steady-state batch=1 cell with
    ``--start-policy auto-warm`` and emit that row without a ``job`` tag; the
    artifact-level start policy distinguishes it from a genuine cold-start
    sample.  Legacy artifacts without a recorded policy are treated as cold.
    """
    if "job" in row:
        return False
    return row.get("_artifact_start_policy", "cold") != "auto-warm"


def _scaling_panel(
    axes: Any,  # noqa: ANN401 - matplotlib axes
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
            and (batch_size != 1 or _cold_batch1_row(row))
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
        axes.set_title("batch size = 128\n(first call cold, then WARM)")
    else:
        axes.set_title("batch size = 1\n(cold start)")
    axes.grid(True, which="both", ls=":", alpha=0.5)
    axes.set_xscale(x_scale)


def _trajectory_panel(
    axes: Any,  # noqa: ANN401 - matplotlib axes
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
    axes.set_title("MD trajectory (WARM)\n(nearly identical frames)")
    axes.grid(True, which="both", ls=":", alpha=0.5)
    axes.set_xscale("log")
    axes.set_yscale("log")


def _strip_svg_trailing_whitespace(path: Path) -> None:
    """Remove trailing whitespace from every SVG line.

    matplotlib's SVG backend writes path data with a trailing space before
    each newline; stripping it keeps the tracked artifact clean for the
    repository's whitespace lint while remaining semantically identical SVG.
    """
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


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
        print(  # noqa: T201 - CLI diagnostics
            "ERROR: at least one --artifact JSON is required", file=sys.stderr
        )
        return 2
    rows, metadata = load_rows(args.artifact)
    commit = args.commit
    if commit is None:
        without_dot_git = metadata.get("commit", {}) or {}
        commit = without_dot_git.get("head") or "unknown"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(  # noqa: T201 - CLI diagnostics
            f"ERROR: matplotlib is required for plotting: {exc}", file=sys.stderr
        )
        return 2

    fig, axes_list = plt.subplots(
        1, 3, figsize=(18, 6), squeeze=False, constrained_layout=True
    )
    _scaling_panel(axes_list[0, 0], rows, 1, args.engines, "log")
    _scaling_panel(axes_list[0, 1], rows, 128, args.engines, "log")
    _trajectory_panel(axes_list[0, 2], rows, args.engines)

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
        "batch = 1 (single molecule) | batch = 128 (128 distinct systems per "
        "batch) | trajectory (WARM)"
    )
    fig.suptitle(title, fontsize=13, y=1.18 if handles else 1.02)
    fig.supxlabel(
        "\n".join(_footnote_lines(metadata, commit)),
        x=0.5,
        fontsize=8,
        family="monospace",
        color="#444444",
    )
    if args.output.suffix == ".svg":
        fig.savefig(args.output, format="svg", bbox_inches="tight")
        _strip_svg_trailing_whitespace(args.output)
    else:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")  # noqa: T201 - CLI completion output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
