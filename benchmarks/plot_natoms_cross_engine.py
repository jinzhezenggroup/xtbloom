#!/usr/bin/env python3
"""Render the gpuxtb cross-engine scaling figure from benchmark artifacts.

The script merges every ``--output-json`` artifact produced by
``natoms_cross_engine.py`` (CPU and CUDA runs, one file per engine set),
keeps only correctness-qualified ``available`` rows, and draws:

1. ``batch=1``: FRESH GFN2-xTB energy+force call latency vs molecule size
   (electronic reset is untimed; gpuxtb lines highlighted);
2. ``batch=128``: WARM latency after an untimed cold seed for 128 *distinct*
   conformers (gpuxtb highlighted);
3. ``batch=512``: FRESH call latency for 512 *distinct* conformers
   (electronic reset is untimed; gpuxtb highlighted).

The figure uses the archived median and min/max range from every eligible row.
Reference engines without a matching point are omitted from that panel; no
invented numbers are inserted.  Detailed hardware, revision, and protocol
metadata stay in the evidence README so the plotting area remains readable at
README and journal-column widths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any


class PlotError(RuntimeError):
    """An inconsistent or ineligible benchmark artifact set."""


def _engine_label(engine: str) -> str:
    labels = {
        "gpuxtb-cpu": "gpuxtb CPU",
        "gpuxtb-cuda": "gpuxtb CUDA†",
        "xtb": "xTB",
        "tblite": "tblite",
        "dxtb-cpu": "dxtb CPU",
        "dxtb-cuda": "dxtb CUDA‡",
    }
    return labels.get(engine, engine)


def _engine_color(engine: str) -> str:
    """Return a colorblind-safe color that keeps gpuxtb visually prominent."""
    colors = {
        "gpuxtb-cpu": "#b2182b",
        "gpuxtb-cuda": "#ef8a62",
        "xtb": "#2166ac",
        "tblite": "#1b9e77",
        "dxtb-cpu": "#7b6fd0",
        "dxtb-cuda": "#56b4e9",
    }
    return colors.get(engine, "#555555")


def _engine_marker(engine: str) -> str:
    """Use redundant marker encoding so the figure survives grayscale output."""
    markers = {
        "gpuxtb-cpu": "o",
        "gpuxtb-cuda": "D",
        "xtb": "s",
        "tblite": "^",
        "dxtb-cpu": "P",
        "dxtb-cuda": "X",
    }
    return markers.get(engine, "o")


def _engine_linestyle(engine: str) -> str:
    """Distinguish CUDA from CPU even when color is unavailable."""
    return "--" if engine.endswith("cuda") else "-"


def load_rows(
    artifact_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge clean, protocol-compatible artifacts without duplicate rows.

    Every row is annotated with ``_artifact_start_policy`` taken from its
    source artifact's ``metadata.protocol.start_policy`` so the batch=1
    cold-start panel can reject job-less rows that were measured under an
    auto-warm start policy (for example the steady-state sample row that a
    trajectory invocation leaks into its output file).
    """
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    common_identity: tuple[Any, ...] | None = None
    row_keys: set[tuple[Any, ...]] = set()
    for path in artifact_paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlotError(f"cannot read artifact {path}: {exc}") from exc
        artifact_metadata = document.get("metadata")
        if document.get("schema_version") != 2 or not isinstance(
            artifact_metadata, dict
        ):
            raise PlotError(f"artifact {path} is not schema-v2 evidence")
        commit = artifact_metadata.get("commit") or {}
        if not commit.get("head") or commit.get("dirty") is not False:
            raise PlotError(f"artifact {path} is not clean-HEAD evidence")
        protocol = artifact_metadata.get("protocol") or {}
        hardware = artifact_metadata.get("hardware") or {}
        threads = artifact_metadata.get("threads") or {}
        identity = (
            commit.get("head"),
            hardware.get("hostname"),
            hardware.get("cpu_model"),
            threads.get("cpu_threads"),
            threads.get("reference_threads"),
            threads.get("dxtb_cpu_threads"),
            protocol.get("warmups"),
            protocol.get("repetitions"),
            protocol.get("cross_engine_energy_atol_hartree"),
            protocol.get("cross_engine_force_atol_hartree_per_bohr"),
            protocol.get("repeatability_energy_atol_hartree"),
            protocol.get("repeatability_force_atol_hartree_per_bohr"),
            protocol.get("scc_max_iterations"),
            protocol.get("scc_charge_tolerance"),
            protocol.get("scc_energy_tolerance"),
        )
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise PlotError(f"artifact {path} uses incompatible run metadata")
        start_policy = protocol.get("start_policy")
        for row in document.get("rows", []):
            key = (
                row.get("engine"),
                row.get("job"),
                row.get("natoms"),
                row.get("batch_size"),
                start_policy,
            )
            if key in row_keys:
                raise PlotError(f"artifact set duplicates row {key}")
            row_keys.add(key)
            row["_artifact_start_policy"] = start_policy
            rows.append(row)
        if not metadata and "metadata" in document:
            metadata = artifact_metadata
    if not rows or not metadata:
        raise PlotError("artifact set contains no rows or metadata")
    return rows, metadata


def _median_ms(row: dict[str, Any]) -> float | None:
    timing = row.get("timing") or {}
    value = timing.get("median_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _timing_range_ms(row: dict[str, Any]) -> tuple[float, float] | None:
    """Return asymmetric min/max distance around the median.

    The retained artifacts contain every raw latency sample.  Plotting their
    full observed range is honest for the small ``n=3`` benchmark protocol and
    avoids implying a population-level confidence interval that was not
    measured.
    """
    median = _median_ms(row)
    samples = (row.get("timing") or {}).get("samples_ms")
    if median is None or not isinstance(samples, list):
        return None
    finite = [
        float(value)
        for value in samples
        if isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    ]
    if not finite:
        return None
    return max(0.0, median - min(finite)), max(0.0, max(finite) - median)


def _is_eligible(row: dict[str, Any]) -> bool:
    """Keep only successful, correctness-qualified, finite rows."""
    if row.get("availability") != "available":
        return False
    median = _median_ms(row)
    if median is None or median <= 0.0 or median != median:
        return False
    correctness = row.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("status") != "pass":
        return False
    cross_engine = correctness.get("cross_engine")
    return isinstance(cross_engine, dict) and cross_engine.get("status") in {
        "pass",
        "reference",
    }


def _hardware_note(metadata: dict[str, Any]) -> str:
    """Build one concise hardware note; full provenance stays in the archive."""
    hardware = metadata.get("hardware", {})
    threads = metadata.get("threads", {})
    pieces: list[str] = []
    cpu_model = hardware.get("cpu_model")
    if cpu_model:
        compact_cpu = str(cpu_model).replace(" 48-Core Processor", "")
        pieces.append(compact_cpu)
        pieces.append(f"{threads.get('cpu_threads', '?')} CPU threads per engine")
    gpu = hardware.get("nvidia_smi")
    if gpu:
        compact_gpu = str(gpu).split("(UUID:", maxsplit=1)[0]
        compact_gpu = compact_gpu.replace("GPU 0:", "").strip()
        pieces.append(compact_gpu)
    return "  ·  ".join(pieces)


def _scientific_notation(value: float) -> str:
    """Format a positive power-of-ten tolerance with Unicode superscripts."""
    if value <= 0.0 or not math.isfinite(value):
        raise PlotError("plot protocol tolerance must be finite and positive")
    exponent = math.log10(value)
    if not math.isclose(exponent, round(exponent), abs_tol=1.0e-10):
        return f"{value:.1e}"
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "10" + str(round(exponent)).translate(superscripts)


def _protocol_note(metadata: dict[str, Any]) -> str:
    """Derive the figure protocol line from validated artifact metadata."""
    protocol = metadata.get("protocol") or {}
    threads = metadata.get("threads") or {}
    repetitions = protocol.get("repetitions")
    cpu_threads = threads.get("cpu_threads")
    charge_tolerance = protocol.get("scc_charge_tolerance")
    energy_tolerance = protocol.get("scc_energy_tolerance")
    if type(repetitions) is not int or repetitions <= 0:
        raise PlotError("plot artifacts have an invalid repetition count")
    if type(cpu_threads) is not int or cpu_threads <= 0:
        raise PlotError("plot artifacts have an invalid CPU thread count")
    if not isinstance(charge_tolerance, (int, float)) or not isinstance(
        energy_tolerance, (int, float)
    ):
        raise PlotError("plot artifacts have incomplete SCC tolerances")
    if math.isclose(float(charge_tolerance), float(energy_tolerance)):
        tolerance_text = _scientific_notation(float(charge_tolerance))
    else:
        tolerance_text = (
            f"q {_scientific_notation(float(charge_tolerance))} / "
            f"E {_scientific_notation(float(energy_tolerance))}"
        )
    return (
        f"Matched nominal SCC tolerance {tolerance_text}  ·  "
        f"{cpu_threads} CPU threads per engine  ·  distinct alkane conformers  ·  "
        f"median of {repetitions} runs"
    )


def _matches_panel_protocol(row: dict[str, Any], batch_size: int) -> bool:
    """Accept only rows measured with the protocol named by one panel.

    dxtb has no warm-continuation public path: it is intentionally accepted in
    panel b only when the row records that its effective policy remained cold.
    """
    expected = {1: "cold", 128: "auto-warm", 512: "cold"}.get(batch_size)
    if expected is None or row.get("job") is not None:
        return False
    artifact_policy = row.get("_artifact_start_policy")
    requested_policy = row.get("start_policy", artifact_policy)
    if artifact_policy != expected or requested_policy != expected:
        return False
    effective_policy = row.get("effective_start_policy")
    if effective_policy is None:
        effective_policy = (
            "cold" if str(row.get("engine", "")).startswith("dxtb") else expected
        )
    if str(row.get("engine", "")).startswith("dxtb"):
        return effective_policy == "cold"
    return effective_policy == expected


def _validate_panel_coverage(rows: list[dict[str, Any]]) -> None:
    """Reject figures with an empty or protocol-mismatched named panel."""
    for batch_size in (1, 128, 512):
        if not any(
            _is_eligible(row)
            and row.get("batch_size") == batch_size
            and _matches_panel_protocol(row, batch_size)
            for row in rows
        ):
            raise PlotError(
                f"artifact set has no eligible protocol-matched batch={batch_size} row"
            )


def _cold_batch1_row(row: dict[str, Any]) -> bool:
    """Compatibility wrapper for focused batch=1 selection tests."""
    return _matches_panel_protocol(row, 1)


def _scaling_panel(
    axes: Any,  # noqa: ANN401 - matplotlib axes
    rows: list[dict[str, Any]],
    batch_size: int,
    engines: list[str],
    panel_label: str,
    title: str,
    subtitle: str,
) -> None:
    """Draw one publication-style scaling panel with gpuxtb highlighted."""
    from matplotlib import ticker

    x_ticks: set[int] = set()
    for engine in engines:
        qualified = [
            row
            for row in rows
            if _is_eligible(row)
            and row.get("engine") == engine
            and row.get("batch_size") == batch_size
            and _matches_panel_protocol(row, batch_size)
        ]
        qualified.sort(key=lambda row: row["natoms"])
        if not qualified:
            continue
        x_values = [float(row["natoms"]) for row in qualified]
        y_values = [_median_ms(row) for row in qualified if _median_ms(row) is not None]
        if len(x_values) != len(y_values):
            continue
        x_ticks.update(int(value) for value in x_values)
        ranges = [_timing_range_ms(row) or (0.0, 0.0) for row in qualified]
        lower = [value[0] for value in ranges]
        upper = [value[1] for value in ranges]
        highlight = engine.startswith("gpuxtb")
        axes.errorbar(
            x_values,
            y_values,
            yerr=[lower, upper],
            marker=_engine_marker(engine),
            markersize=(4.8 if highlight else 3.6),
            markeredgewidth=(0.8 if highlight else 0.5),
            markeredgecolor=("white" if highlight else _engine_color(engine)),
            linewidth=(1.75 if highlight else 1.05),
            linestyle=_engine_linestyle(engine),
            color=_engine_color(engine),
            alpha=(1.0 if highlight else 0.88),
            elinewidth=(0.7 if highlight else 0.5),
            capsize=1.5,
            capthick=0.5,
            zorder=(5 if highlight else 2),
            label=_engine_label(engine),
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    ordered_ticks = sorted(x_ticks)
    crowded = any(right / left < 1.35 for left, right in pairwise(ordered_ticks))
    axes.set_xticks(ordered_ticks)
    axes.set_xticklabels(
        [str(value) for value in ordered_ticks],
        rotation=(32 if crowded else 0),
        ha=("right" if crowded else "center"),
    )
    axes.xaxis.set_minor_locator(ticker.NullLocator())
    axes.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(_format_latency_tick))
    axes.yaxis.set_minor_formatter(ticker.NullFormatter())
    axes.set_xlabel("Atoms per system", labelpad=6)
    axes.set_title(
        f"{title}\n{subtitle}",
        loc="left",
        pad=9,
        fontsize=8.0,
        fontweight="semibold",
        color="#171a1f",
    )
    axes.text(
        -0.11,
        1.16,
        panel_label,
        transform=axes.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        color="#171a1f",
    )
    axes.set_axisbelow(True)
    axes.grid(True, axis="y", which="major", color="#d8dce2", linewidth=0.7)
    axes.grid(True, axis="y", which="minor", color="#eef0f3", linewidth=0.45)
    axes.grid(True, axis="x", which="major", color="#eef0f3", linewidth=0.45)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axes.spines[side].set_color("#59616b")
        axes.spines[side].set_linewidth(0.75)
    axes.tick_params(
        axis="both",
        which="both",
        direction="out",
        colors="#30343b",
        width=0.7,
        labelsize=6.8,
    )


def _format_latency_tick(value: float, _position: float) -> str:
    """Format log-scale latency ticks as 10, 100, 1k, 10k, ... ."""
    if value <= 0.0:
        return ""
    exponent = math.log10(value)
    if not math.isclose(exponent, round(exponent), abs_tol=1.0e-8):
        return ""
    if value >= 1.0e6:
        return f"{value / 1.0e6:g}M"
    if value >= 1.0e3:
        return f"{value / 1.0e3:g}k"
    return f"{value:g}"


def _speedup_range(
    rows: list[dict[str, Any]],
    batch_size: int,
    natoms: int,
) -> tuple[float, float, float, float] | None:
    """Return target latency, nearest reference, and xTB/tblite speedup range."""
    values: dict[str, float] = {}
    for row in rows:
        engine = row.get("engine")
        if (
            engine in {"gpuxtb-cpu", "xtb", "tblite"}
            and row.get("batch_size") == batch_size
            and row.get("natoms") == natoms
            and _is_eligible(row)
            and _matches_panel_protocol(row, batch_size)
        ):
            median = _median_ms(row)
            if median is not None:
                values[str(engine)] = median
    if not {"gpuxtb-cpu", "xtb", "tblite"}.issubset(values):
        return None
    target = values["gpuxtb-cpu"]
    references = [values["xtb"], values["tblite"]]
    ratios = [reference / target for reference in references]
    return target, min(references), min(ratios), max(ratios)


def _annotate_speedup(
    axes: Any,  # noqa: ANN401 - matplotlib axes
    rows: list[dict[str, Any]],
    batch_size: int,
    natoms: int = 62,
) -> None:
    """Draw a data-derived speedup bracket at one shared benchmark coordinate."""
    summary = _speedup_range(rows, batch_size, natoms)
    if summary is None:
        return
    target, nearest_reference, minimum, maximum = summary
    axes.annotate(
        "",
        xy=(natoms, target),
        xytext=(natoms, nearest_reference),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#555d66",
            "linewidth": 0.9,
            "shrinkA": 5,
            "shrinkB": 5,
        },
        zorder=7,
    )
    if maximum < 2.0:
        speedup = f"{minimum:.1f}-{maximum:.1f}x"
    else:
        speedup = f"{minimum:.0f}-{maximum:.0f}x"
    axes.annotate(
        f"{speedup} faster\nvs xTB / tblite",
        xy=(natoms, math.sqrt(target * nearest_reference)),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="semibold",
        color="#343a40",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#d8dce2",
            "linewidth": 0.65,
            "alpha": 0.94,
        },
        zorder=8,
    )


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Merge artifacts and write the three-panel figure."""
    args = build_parser().parse_args(argv)
    if not args.artifact:
        print(  # noqa: T201 - CLI diagnostics
            "ERROR: at least one --artifact JSON is required", file=sys.stderr
        )
        return 2
    try:
        rows, metadata = load_rows(args.artifact)
        _validate_panel_coverage(rows)
        protocol_note = _protocol_note(metadata)
    except PlotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 2
    commit = (metadata.get("commit") or {}).get("head") or "unknown"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(  # noqa: T201 - CLI diagnostics
            f"ERROR: matplotlib is required for plotting: {exc}", file=sys.stderr
        )
        return 2

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelcolor": "#30343b",
            "text.color": "#30343b",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), squeeze=True)
    fig.subplots_adjust(left=0.08, right=0.992, bottom=0.18, top=0.67, wspace=0.27)
    _scaling_panel(
        axes[0],
        rows,
        1,
        args.engines,
        "a",
        "Single molecule",
        "batch = 1  ·  FRESH (reset untimed)",
    )
    _scaling_panel(
        axes[1],
        rows,
        128,
        args.engines,
        "b",
        "Conformer throughput",
        "batch = 128  ·  WARM after seed (dxtb cold)",
    )
    _scaling_panel(
        axes[2],
        rows,
        512,
        args.engines,
        "c",
        "Large-batch throughput",
        "batch = 512  ·  FRESH (reset untimed)",
    )
    axes[0].set_ylabel("Latency per call (ms)", labelpad=7)
    for axes_item, batch_size in zip(axes, (1, 128, 512), strict=True):
        _annotate_speedup(axes_item, rows, batch_size)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.077, 0.814),
            borderaxespad=0.0,
            handlelength=2.1,
            handletextpad=0.4,
            columnspacing=0.9,
            fontsize=6.7,
        )
    fig.text(
        0.08,
        0.955,
        "GFN2-xTB energy + force latency",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#171a1f",
    )
    fig.text(
        0.08,
        0.895,
        protocol_note,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#59616b",
    )
    fig.text(
        0.08,
        0.035,
        _hardware_note(metadata)
        + f"  ·  runner {commit[:8]}  ·  whiskers: min-max  ·  lower is better"
        + "  ·  † host descriptor  ·  ‡ device-resident input",
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="#69717b",
    )
    if args.output.suffix == ".svg":
        fig.savefig(args.output, format="svg", facecolor="white")
        _strip_svg_trailing_whitespace(args.output)
    else:
        fig.savefig(args.output, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")  # noqa: T201 - CLI completion output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
