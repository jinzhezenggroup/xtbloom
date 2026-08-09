#!/usr/bin/env python3
"""Render the gpuxtb cross-engine scaling figure from benchmark artifacts.

The script merges every ``--output-json`` artifact produced by
``natoms_cross_engine.py`` (CPU and CUDA runs, one file per engine set),
keeps only correctness-qualified ``available`` rows, and draws:

1. ``batch=1``: FRESH GFN2-xTB energy+force public-call latency vs molecule
   size (engine-specific state preparation stays explicit in the evidence);
2. ``batch=128``: WARM latency after an untimed cold seed for 128 *distinct*
   conformers (gpuxtb highlighted);
3. ``batch=512``: FRESH public-call latency for 512 *distinct* conformers
   (gpuxtb highlighted).

The figure uses the archived median and min/max range from every eligible row.
Reference engines without a matching point are omitted from that panel; no
invented numbers are inserted.  Detailed hardware, revision, and protocol
metadata stay in the evidence README so the plotting area remains readable at
README and journal-column widths.
"""

from __future__ import annotations

import argparse
import hashlib
import html
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
        "dxtb-cpu": "#6a51a3",
        "dxtb-cuda": "#b07aa1",
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


def _native_fingerprint(identity: dict[str, Any] | None) -> dict[str, Any]:
    """Return path-independent native-library and dependency hashes."""
    identity = identity or {}
    return {
        "sha256": identity.get("sha256"),
        "dependencies": [
            (item.get("soname"), item.get("sha256"))
            for item in identity.get("resolved_dependencies") or []
        ],
        "unresolved_dependencies": identity.get("unresolved_dependencies") or [],
    }


def _cuda_device_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the runtime-verified stable selected GPU identity."""
    selected = (metadata.get("hardware") or {}).get("selected_cuda_device") or {}
    device = selected.get("device") or {}
    identity = {
        name: device.get(name) for name in ("uuid", "name", "driver", "memory_mib")
    }
    if not identity.get("uuid") or selected.get("runtime_uuid") != identity.get("uuid"):
        raise PlotError("artifact has no runtime-verified selected GPU")
    return identity


def _engine_runtime_identity(metadata: dict[str, Any], engine: str) -> str:
    """Build one stable runtime fingerprint for cross-artifact consistency."""
    runner = metadata.get("runner") or {}
    if engine.startswith("gpuxtb"):
        build = runner.get("gpuxtb_build") or {}
        source = build.get("source_state") or {}
        identity = {
            "library_sha256": runner.get("gpuxtb_library_sha256"),
            "source_head": source.get("head"),
            "source_dirty": source.get("dirty"),
            "cmake_selected": build.get("selected"),
            "native": _native_fingerprint(runner.get("gpuxtb_native_identity")),
        }
    elif engine in {"xtb", "tblite"}:
        source = runner.get(f"{engine}_source") or {}
        identity = {
            "library_sha256": runner.get(f"{engine}_library_sha256"),
            "source_head": source.get("head"),
            "source_dirty": source.get("dirty"),
            "native": _native_fingerprint(runner.get(f"{engine}_native_identity")),
        }
    elif engine.startswith("dxtb"):
        source = runner.get("dxtb_source") or {}
        distributions = runner.get("python_distributions") or {}
        identity = {
            "source_head": source.get("head"),
            "source_dirty": source.get("dirty"),
            "distributions": {
                name: {
                    "version": (distributions.get(name) or {}).get("version"),
                    "payload": (
                        (distributions.get(name) or {}).get("payload_verification")
                        or {}
                    ).get("payload_sha256"),
                    "status": (
                        (distributions.get(name) or {}).get("payload_verification")
                        or {}
                    ).get("status"),
                    "direct_url_identity": (distributions.get(name) or {}).get(
                        "direct_url_identity"
                    ),
                }
                for name in ("dxtb", "torch", "tad-libcint")
            },
        }
    else:
        raise PlotError(f"unsupported engine runtime identity: {engine}")
    if engine.endswith("cuda"):
        identity["cuda_device"] = _cuda_device_identity(metadata)
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    if not identity.get("source_head") or identity.get("source_dirty") is not False:
        raise PlotError(f"artifact has incomplete clean source identity for {engine}")
    if engine.startswith("dxtb"):
        if any(
            item.get("status") != "verified" or not item.get("payload")
            for item in identity["distributions"].values()
        ):
            raise PlotError(
                f"artifact has incomplete installed payload identity for {engine}"
            )
    elif not identity.get("library_sha256") or not identity["native"].get("sha256"):
        raise PlotError(f"artifact has incomplete native identity for {engine}")
    return serialized


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
    engine_identities: dict[str, str] = {}
    common_cuda_device: str | None = None
    independent_artifacts: set[tuple[str, str, str]] = set()
    for path in artifact_paths:
        try:
            payload = path.read_bytes()
            document = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlotError(f"cannot read artifact {path}: {exc}") from exc
        artifact_sha256 = hashlib.sha256(payload).hexdigest()
        artifact_metadata = document.get("metadata")
        if document.get("schema_version") != 2 or not isinstance(
            artifact_metadata, dict
        ):
            raise PlotError(f"artifact {path} is not schema-v2 evidence")
        commit = artifact_metadata.get("commit") or {}
        if not commit.get("head") or commit.get("dirty") is not False:
            raise PlotError(f"artifact {path} is not clean-HEAD evidence")
        eligibility = artifact_metadata.get("evidence_eligibility") or {}
        if (
            eligibility.get("status") != "eligible_clean_head"
            or eligibility.get("allow_dirty_evidence") is not False
        ):
            raise PlotError(f"artifact {path} used a diagnostic evidence override")
        protocol = artifact_metadata.get("protocol") or {}
        hardware = artifact_metadata.get("hardware") or {}
        threads = artifact_metadata.get("threads") or {}
        reference = artifact_metadata.get("comparison_reference") or {}
        identity = (
            commit.get("head"),
            hardware.get("hostname"),
            hardware.get("cpu_model"),
            tuple(hardware.get("process_cpu_affinity") or ()),
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
            json.dumps(
                protocol.get("convergence_contract"),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise PlotError(f"artifact {path} uses incompatible run metadata")
        start_policy = protocol.get("start_policy")
        if reference.get("designation") == "independent_baseline":
            reference_engine = reference.get("engine")
            if reference_engine not in {"xtb", "tblite"}:
                raise PlotError(f"artifact {path} has an invalid independent baseline")
            independent_artifacts.add(
                (str(start_policy), str(reference_engine), artifact_sha256)
            )
        for row in document.get("rows", []):
            engine = str(row.get("engine"))
            key = (
                engine,
                row.get("job"),
                row.get("natoms"),
                row.get("batch_size"),
                start_policy,
            )
            if key in row_keys:
                raise PlotError(f"artifact set duplicates row {key}")
            row_keys.add(key)
            row["_artifact_start_policy"] = start_policy
            row["_artifact_sha256"] = artifact_sha256
            row["_artifact_reference"] = reference
            runtime_identity = _engine_runtime_identity(artifact_metadata, engine)
            previous_identity = engine_identities.setdefault(engine, runtime_identity)
            if previous_identity != runtime_identity:
                raise PlotError(
                    f"artifact set changes the runtime identity for engine {engine}"
                )
            if engine.endswith("cuda"):
                cuda_device = json.dumps(
                    _cuda_device_identity(artifact_metadata),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if common_cuda_device is None:
                    common_cuda_device = cuda_device
                elif cuda_device != common_cuda_device:
                    raise PlotError(
                        "artifact set mixes different selected GPUs across CUDA engines"
                    )
            rows.append(row)
        if not metadata and "metadata" in document:
            metadata = artifact_metadata
    if not rows or not metadata:
        raise PlotError("artifact set contains no rows or metadata")
    for row in rows:
        correctness = row.get("correctness") or {}
        comparison = correctness.get("cross_engine") or {}
        status = comparison.get("status")
        reference = row.get("_artifact_reference") or {}
        if status == "reference":
            if reference.get("designation") != "independent_baseline" or reference.get(
                "engine"
            ) != row.get("engine"):
                raise PlotError("reference row is not carried by its baseline artifact")
        elif status == "pass":
            expected = (
                str(row.get("_artifact_start_policy")),
                str(comparison.get("reference_engine")),
                str(comparison.get("artifact_sha256")),
            )
            if (
                reference.get("designation") != "dependent_run"
                or reference.get("engine") != comparison.get("reference_engine")
                or reference.get("artifact_sha256") != comparison.get("artifact_sha256")
                or expected not in independent_artifacts
            ):
                raise PlotError(
                    "dependent row does not link to a supplied panel-matched reference"
                )
    return rows, metadata


def _median_ms(row: dict[str, Any]) -> float | None:
    timing = row.get("timing") or {}
    value = timing.get("median_ms")
    return float(value) if isinstance(value, int | float) else None


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
        if isinstance(value, int | float)
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


def _scientific_notation(value: float) -> str:
    """Format a positive tolerance with a compact Unicode power of ten."""
    if value <= 0.0 or not math.isfinite(value):
        raise PlotError("plot protocol tolerance must be finite and positive")
    exponent = math.floor(math.log10(value))
    coefficient = value / (10.0**exponent)
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    power = "10" + str(exponent).translate(superscripts)
    if math.isclose(coefficient, 1.0, abs_tol=1.0e-10):
        return power
    return f"{coefficient:g}\N{MULTIPLICATION SIGN}{power}"


def _protocol_note(metadata: dict[str, Any]) -> str:
    """Derive two concise method lines from validated artifact metadata."""
    protocol = metadata.get("protocol") or {}
    threads = metadata.get("threads") or {}
    repetitions = protocol.get("repetitions")
    cpu_threads = threads.get("cpu_threads")
    charge_tolerance = protocol.get("scc_charge_tolerance")
    energy_tolerance = protocol.get("scc_energy_tolerance")
    output_energy_atol = protocol.get("cross_engine_energy_atol_hartree")
    output_force_atol = protocol.get("cross_engine_force_atol_hartree_per_bohr")
    if type(repetitions) is not int or repetitions <= 0:
        raise PlotError("plot artifacts have an invalid repetition count")
    if type(cpu_threads) is not int or cpu_threads <= 0:
        raise PlotError("plot artifacts have an invalid CPU thread count")
    if not isinstance(charge_tolerance, int | float) or not isinstance(
        energy_tolerance, int | float
    ):
        raise PlotError("plot artifacts have incomplete SCC tolerances")
    if not isinstance(output_energy_atol, int | float) or not isinstance(
        output_force_atol, int | float
    ):
        raise PlotError("plot artifacts have incomplete output compatibility gates")
    if math.isclose(float(charge_tolerance), float(energy_tolerance)):
        tolerance_text = _scientific_notation(float(charge_tolerance))
    else:
        tolerance_text = (
            f"q {_scientific_notation(float(charge_tolerance))} / "
            f"E {_scientific_notation(float(energy_tolerance))}"
        )
    contract = protocol.get("convergence_contract") or {}
    xtb_accuracy = (contract.get("xtb") or {}).get("public_accuracy_factor")
    tblite_accuracy = (contract.get("tblite") or {}).get("public_accuracy_factor")
    dxtb_contract = contract.get("dxtb") or {}
    if (
        not all(
            isinstance(value, int | float)
            for value in (
                xtb_accuracy,
                tblite_accuracy,
                dxtb_contract.get("x_atol"),
                dxtb_contract.get("x_atol_max"),
                dxtb_contract.get("f_atol"),
            )
        )
        or dxtb_contract.get("force_convergence") is not True
    ):
        raise PlotError("plot artifacts have an incomplete convergence contract")
    if math.isclose(float(xtb_accuracy), float(tblite_accuracy)):
        reference_text = f"xTB/tblite accuracy {float(xtb_accuracy):g}"
    else:
        reference_text = (
            f"xTB accuracy {float(xtb_accuracy):g} / "
            f"tblite accuracy {float(tblite_accuracy):g}"
        )
    dxtb_text = (
        f"dxtb x {_scientific_notation(float(dxtb_contract['x_atol']))}/"
        f"{_scientific_notation(float(dxtb_contract['x_atol_max']))}, "
        f"f {_scientific_notation(float(dxtb_contract['f_atol']))}"
    )
    return (
        "Native controls (not identical): "
        f"gpuxtb {tolerance_text} · {reference_text} · {dxtb_text}\n"
        f"CPU rows: {cpu_threads} threads · median n={repetitions}, "
        f"min\N{EN DASH}max · eligibility |ΔE| ≤ "
        f"{_scientific_notation(float(output_energy_atol))} Eh; "
        f"maxᵢ|ΔFᵢ| ≤ {_scientific_notation(float(output_force_atol))} Eh bohr⁻¹ "
        "(not conformance)"
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
    """Draw one publication-style panel without bridging failed coordinates."""
    from matplotlib import ticker

    x_ticks: set[int] = set()
    for engine in engines:
        requested = [
            row
            for row in rows
            if row.get("engine") == engine
            and row.get("batch_size") == batch_size
            and _matches_panel_protocol(row, batch_size)
        ]
        requested.sort(key=lambda row: row["natoms"])
        if not requested:
            continue
        x_ticks.update(int(row["natoms"]) for row in requested)
        highlight = engine.startswith("gpuxtb")
        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in requested:
            if _is_eligible(row):
                current.append(row)
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        labeled = False
        for segment in segments:
            x_values = [float(row["natoms"]) for row in segment]
            y_values = [float(_median_ms(row)) for row in segment]
            ranges = [_timing_range_ms(row) or (0.0, 0.0) for row in segment]
            lower = [value[0] for value in ranges]
            upper = [value[1] for value in ranges]
            axes.errorbar(
                x_values,
                y_values,
                yerr=[lower, upper],
                marker=_engine_marker(engine),
                markersize=(5.4 if highlight else 4.4),
                markeredgewidth=(0.9 if highlight else 0.7),
                markeredgecolor=("white" if highlight else _engine_color(engine)),
                linewidth=(1.9 if highlight else 1.2),
                linestyle=_engine_linestyle(engine),
                color=_engine_color(engine),
                alpha=(1.0 if highlight else 0.9),
                elinewidth=(0.95 if highlight else 0.75),
                capsize=2.2,
                capthick=0.75,
                zorder=(5 if highlight else 2),
                label=_engine_label(engine) if not labeled else None,
            )
            labeled = True
        for row in requested:
            if _is_eligible(row):
                continue
            natoms = float(row["natoms"])
            median = _median_ms(row)
            if (
                row.get("availability") == "available"
                and median is not None
                and math.isfinite(median)
                and median > 0.0
            ):
                # The latency is real, but this row failed the declared output
                # compatibility gate.  A hollow marker preserves the timing
                # without allowing it to support a speed claim or bridge a line.
                axes.plot(
                    [natoms],
                    [median],
                    linestyle="none",
                    marker=_engine_marker(engine),
                    markersize=(5.4 if highlight else 4.4),
                    markerfacecolor="white",
                    markeredgecolor=_engine_color(engine),
                    markeredgewidth=1.1,
                    alpha=0.9,
                    zorder=4,
                    label=_engine_label(engine) if not labeled else None,
                )
            else:
                axes.plot(
                    [natoms],
                    [0.035],
                    transform=axes.get_xaxis_transform(),
                    linestyle="none",
                    marker="x",
                    markersize=4.8,
                    markeredgewidth=1.0,
                    color=_engine_color(engine),
                    clip_on=False,
                    zorder=6,
                    label=_engine_label(engine) if not labeled else None,
                )
                detail = str(row.get("error") or row.get("reason") or "unavailable")
                label = "OOM" if "memory" in detail.casefold() else "unavailable"
                axes.annotate(
                    label,
                    xy=(natoms, 0.035),
                    xycoords=axes.get_xaxis_transform(),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color=_engine_color(engine),
                    clip_on=False,
                )
            labeled = True
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
    axes.text(
        0.0,
        1.13,
        panel_label,
        transform=axes.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        color="#171a1f",
    )
    axes.text(
        0.105,
        1.13,
        title,
        transform=axes.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="bottom",
        color="#171a1f",
    )
    axes.text(
        0.0,
        1.045,
        subtitle,
        transform=axes.transAxes,
        fontsize=7.5,
        va="bottom",
        color="#626a73",
    )
    axes.set_axisbelow(True)
    axes.grid(True, axis="y", which="major", color="#d8dce2", linewidth=0.7)
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
        labelsize=7.6,
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
) -> tuple[float, float, float, float, str] | None:
    """Return gpuxtb CPU latency and qualified baseline ratio range."""
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
    if "gpuxtb-cpu" not in values:
        return None
    target = values["gpuxtb-cpu"]
    reference_names = [name for name in ("xtb", "tblite") if name in values]
    if not reference_names:
        return None
    references = [values[name] for name in reference_names]
    ratios = [reference / target for reference in references]
    labels = {"xtb": "xTB", "tblite": "tblite"}
    comparison = " / ".join(labels[name] for name in reference_names)
    return target, max(references), min(ratios), max(ratios), comparison


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
    target, farthest_reference, minimum, maximum, comparison = summary
    axes.annotate(
        "",
        xy=(natoms, target),
        xytext=(natoms, farthest_reference),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#555d66",
            "linewidth": 0.9,
            "shrinkA": 5,
            "shrinkB": 5,
        },
        zorder=7,
    )
    if math.isclose(minimum, maximum, rel_tol=0.02):
        speedup = (
            f"{minimum:.1f}\N{MULTIPLICATION SIGN}"
            if minimum < 2.0
            else f"{minimum:.0f}\N{MULTIPLICATION SIGN}"
        )
    elif maximum < 2.0:
        speedup = f"{minimum:.1f}\N{EN DASH}{maximum:.1f}\N{MULTIPLICATION SIGN}"
    else:
        speedup = f"{minimum:.0f}\N{EN DASH}{maximum:.0f}\N{MULTIPLICATION SIGN}"
    axes.annotate(
        f"{speedup} lower latency\nvs {comparison}",
        xy=(natoms, math.sqrt(target * farthest_reference)),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="medium",
        color="#343a40",
        bbox={
            "boxstyle": "square,pad=0.18",
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


def _add_svg_accessibility(path: Path, title: str, description: str) -> None:
    """Insert deterministic SVG title/description elements for screen readers."""
    text = path.read_text(encoding="utf-8")
    svg_start = text.find("<svg")
    if svg_start < 0:
        raise PlotError("matplotlib output has no SVG root element")
    opening_end = text.find(">", svg_start)
    if opening_end < 0:
        raise PlotError("matplotlib output has an incomplete SVG root element")
    accessibility = (
        f"\n <title>{html.escape(title)}</title>"
        f"\n <desc>{html.escape(description)}</desc>"
    )
    path.write_text(
        text[: opening_end + 1] + accessibility + text[opening_end + 1 :],
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
            "font.sans-serif": [
                "Liberation Sans",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 8.0,
            "axes.labelcolor": "#30343b",
            "text.color": "#30343b",
            "svg.fonttype": "none",
            "svg.hashsalt": "gpuxtb-natoms-cross-engine-v2",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9), squeeze=True, sharey=True)
    fig.subplots_adjust(left=0.082, right=0.992, bottom=0.17, top=0.6, wspace=0.2)
    _scaling_panel(
        axes[0],
        rows,
        1,
        args.engines,
        "a",
        "Single-system latency",
        "batch 1 · FRESH public call",
    )
    _scaling_panel(
        axes[1],
        rows,
        128,
        args.engines,
        "b",
        "128-system batch",
        "distinct conformers · WARM after untimed seed; dxtb cold",
    )
    _scaling_panel(
        axes[2],
        rows,
        512,
        args.engines,
        "c",
        "512-system batch",
        "distinct conformers · FRESH public call",
    )
    axes[0].set_ylabel("Latency per call (ms)", labelpad=7)
    for axes_item, batch_size in zip(axes[1:], (128, 512), strict=True):
        _annotate_speedup(axes_item, rows, batch_size)

    legend_items: dict[str, Any] = {}
    for axes_item in axes:
        handles, labels = axes_item.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            legend_items.setdefault(label, handle)
    from matplotlib.lines import Line2D

    panel_rows = [
        row
        for row in rows
        if row.get("batch_size") in {1, 128, 512}
        and _matches_panel_protocol(row, int(row["batch_size"]))
    ]
    if any(
        row.get("availability") == "available"
        and _median_ms(row) is not None
        and not _is_eligible(row)
        for row in panel_rows
    ):
        legend_items["Failed eligibility gate"] = Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor="#59616b",
        )
    if any(row.get("availability") != "available" for row in panel_rows):
        legend_items["Unavailable / OOM"] = Line2D(
            [],
            [],
            linestyle="none",
            marker="x",
            markersize=4.6,
            markeredgecolor="#59616b",
        )
    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items),
            loc="upper left",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.078, 0.8),
            borderaxespad=0.0,
            handlelength=2.3,
            handletextpad=0.5,
            columnspacing=1.25,
            labelspacing=0.55,
            fontsize=7.3,
        )
    fig.text(
        0.08,
        0.975,
        "GFN2-xTB energy + analytic-force latency",
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#171a1f",
    )
    fig.text(
        0.08,
        0.915,
        protocol_note,
        ha="left",
        va="top",
        fontsize=7.1,
        color="#59616b",
        linespacing=1.35,
    )
    fig.text(
        0.082,
        0.035,
        "Engine-specific state boundaries are detailed in the caption · "
        "CUDA: † host descriptors, ‡ device tensors; not directly comparable.",
        ha="left",
        va="bottom",
        fontsize=6.7,
        color="#69717b",
    )
    if args.output.suffix == ".svg":
        fig.savefig(
            args.output,
            format="svg",
            facecolor="white",
            metadata={
                "Date": None,
                "Creator": "gpuxtb plot_natoms_cross_engine.py",
            },
        )
        _add_svg_accessibility(
            args.output,
            "Cross-engine GFN2-xTB energy and analytic-force latency",
            "Three log-scale panels compare batch 1, 128, and 512 systems. "
            "Filled points pass the declared output compatibility gate; hollow "
            "points fail it, and x marks unavailable or errored coordinates.",
        )
        _strip_svg_trailing_whitespace(args.output)
    else:
        fig.savefig(args.output, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")  # noqa: T201 - CLI completion output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
