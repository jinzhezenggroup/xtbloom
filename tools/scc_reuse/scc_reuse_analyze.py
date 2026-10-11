#!/usr/bin/env python3
"""Analyze SCC occupied-subspace reuse for issue #343 Phase 1.

Within one geometry, all metrics use the ordinary AO overlap ``S``.  Across
geometries, physical principal angles use the explicitly captured cross-AO
overlap ``S12 = <chi(R1)|chi(R2)>``.  Algorithmic residuals use a separate,
documented co-moving AO-label transport: source coefficients are interpreted
on the target AO labels, orthonormalized in ``S2``, Rayleigh--Ritz rotated with
the target terminal effective SCC Hamiltonian, and then tested against that
same generalized eigenproblem.

The capture format is experimental and deliberately separate from the pinned
``xtbloom-scc-trace-v1`` conformance contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Record = dict[str, Any]
_SINGULAR_VALUE_TOLERANCE = 1.0e-8
_GRAM_EIGENVALUE_TOLERANCE = 1.0e-10


class DocumentError(ValueError):
    """Report a malformed or scientifically unsupported diagnostic stream."""


class Parser:
    """Consume the line-oriented ``xtbloom-scc-reuse-v2`` format."""

    def __init__(self, path: Path) -> None:
        """Load non-empty ASCII lines from ``path``."""
        with path.open(encoding="ascii") as handle:
            self.lines = [line.strip() for line in handle if line.strip()]
        self.index = 0

    def next(self) -> str:
        """Return the next line or raise a structured truncation error."""
        if self.index >= len(self.lines):
            raise DocumentError("unexpected end of document")
        line = self.lines[self.index]
        self.index += 1
        return line

    def peek(self) -> str | None:
        """Return the next line without consuming it."""
        if self.index >= len(self.lines):
            return None
        return self.lines[self.index]

    def expect(self, label: str) -> None:
        """Consume an exact label."""
        line = self.next()
        if line != label:
            raise DocumentError(f"line {self.index}: expected {label!r}, got {line!r}")

    def scalar(self, label: str) -> str:
        """Read a scalar written either beside or below ``label``."""
        line = self.next()
        if line == label:
            return self.next()
        if line.startswith(label + " "):
            return line[len(label) + 1 :]
        raise DocumentError(f"line {self.index}: expected {label!r}, got {line!r}")

    def int_value(self, label: str) -> int:
        """Read an integer scalar."""
        try:
            return int(self.scalar(label))
        except ValueError as exc:
            raise DocumentError(
                f"line {self.index}: expected integer for {label!r}"
            ) from exc

    def float_value(self, label: str) -> float:
        """Read a floating-point scalar."""
        try:
            return float(self.scalar(label))
        except ValueError as exc:
            raise DocumentError(
                f"line {self.index}: expected float for {label!r}"
            ) from exc

    def ints(self, count: int) -> list[int]:
        """Read ``count`` integer lines."""
        try:
            return [int(self.next()) for _ in range(count)]
        except ValueError as exc:
            raise DocumentError(f"line {self.index}: expected integer") from exc

    def floats(self, count: int) -> list[float]:
        """Read ``count`` floating-point lines."""
        try:
            return [float(self.next()) for _ in range(count)]
        except ValueError as exc:
            raise DocumentError(f"line {self.index}: expected float") from exc


def matrix(parser: Parser, size: int) -> FloatArray:
    """Read one row-major square binary64 matrix."""
    return np.asarray(parser.floats(size * size), dtype=np.float64).reshape(size, size)


def parse_document(path: Path) -> Record:
    """Parse and validate one restricted ``xtbloom-scc-reuse-v2`` stream."""
    parser = Parser(path)
    header = parser.next()
    if header != "diagnostic xtbloom-scc-reuse-v2":
        raise DocumentError(f"{path}: not an xtbloom-scc-reuse-v2 document")

    def read_case() -> Record:
        line = parser.next()
        if not line.startswith("case "):
            raise DocumentError(
                f"line {parser.index}: expected 'case <name>', got {line!r}"
            )
        atom_count = parser.int_value("nat")
        parser.expect("atomic_numbers")
        atomic_numbers = parser.ints(atom_count)
        parser.expect("positions")
        positions = parser.floats(3 * atom_count)
        return {
            "case": line[len("case ") :],
            "nat": atom_count,
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "molecular_charge": parser.float_value("molecular_charge"),
            "unpaired_electrons": parser.int_value("unpaired_electrons"),
            "temperature_kelvin": parser.float_value("temperature_kelvin"),
            "mixer_memory": parser.int_value("mixer_memory"),
            "mixer_damping": parser.float_value("mixer_damping"),
            "maximum_iterations": parser.int_value("maximum_iterations"),
        }

    def read_geometry(case: Record) -> Record:
        generation = parser.int_value("geometry")
        role = parser.scalar("trajectory_role")
        start_mode = parser.scalar("start_mode")
        basis_size = parser.int_value("nao")
        spin_channels = parser.int_value("nspin")
        if spin_channels != 1:
            raise DocumentError(
                "issue #343 Phase-1 analyzer supports restricted captures only"
            )
        parser.expect("overlap")
        overlap = matrix(parser, basis_size)
        parser.expect("core_hamiltonian")
        core_hamiltonian = matrix(parser, basis_size)
        has_cross_overlap = parser.int_value("has_cross_overlap")
        if has_cross_overlap not in (0, 1):
            raise DocumentError("has_cross_overlap must be zero or one")
        cross_overlap = None
        if has_cross_overlap:
            parser.expect("cross_overlap_from_source")
            cross_overlap = matrix(parser, basis_size)

        iterations: list[Record] = []
        while True:
            line = parser.next()
            if line.startswith("iteration "):
                iteration = int(line[len("iteration ") :])
                step_micros = parser.int_value("step_micros")
                eigensolve_micros = parser.int_value("eigensolve_micros")
                parser.expect("hamiltonian")
                hamiltonian = matrix(parser, basis_size)
                parser.expect("coefficients")
                coefficients = np.asarray(
                    parser.floats(spin_channels * basis_size * basis_size),
                    dtype=np.float64,
                ).reshape(spin_channels, basis_size, basis_size)
                parser.expect("eigenvalues")
                eigenvalues = np.asarray(
                    parser.floats(spin_channels * basis_size), dtype=np.float64
                ).reshape(spin_channels, basis_size)
                parser.expect("occupations")
                occupations = np.asarray(
                    parser.floats(2 * basis_size), dtype=np.float64
                ).reshape(2, basis_size)
                parser.expect("density")
                density = matrix(parser, basis_size)
                iterations.append(
                    {
                        "k": iteration,
                        "step_micros": step_micros,
                        "eigensolve_micros": eigensolve_micros,
                        "H": hamiltonian,
                        "C": coefficients,
                        "eps": eigenvalues,
                        "occ": occupations,
                        "P": density,
                    }
                )
                continue
            if line != "converged":
                raise DocumentError(f"unexpected token {line!r} inside geometry body")
            parser.expect("terminal_effective_hamiltonian")
            terminal_hamiltonian = matrix(parser, basis_size)
            parser.expect("coefficients")
            coefficients = np.asarray(
                parser.floats(spin_channels * basis_size * basis_size),
                dtype=np.float64,
            ).reshape(spin_channels, basis_size, basis_size)
            parser.expect("eigenvalues")
            eigenvalues = np.asarray(
                parser.floats(spin_channels * basis_size), dtype=np.float64
            ).reshape(spin_channels, basis_size)
            parser.expect("occupations")
            occupations = np.asarray(
                parser.floats(2 * basis_size), dtype=np.float64
            ).reshape(2, basis_size)
            parser.expect("density")
            density = matrix(parser, basis_size)
            return {
                "case": case,
                "generation": generation,
                "role": role,
                "start_mode": start_mode,
                "nao": basis_size,
                "nspin": spin_channels,
                "overlap": overlap,
                "core_hamiltonian": core_hamiltonian,
                "cross_overlap_from_source": cross_overlap,
                "iterations": iterations,
                "converged": {
                    "H": terminal_hamiltonian,
                    "C": coefficients,
                    "eps": eigenvalues,
                    "occ": occupations,
                    "P": density,
                },
                "converged_state": parser.int_value("converged_state") == 1,
            }

    cases: list[Record] = []
    geometries: list[Record] = []
    current_case: Record | None = None
    terminal_line = ""
    while True:
        line = parser.peek()
        if line is None:
            raise DocumentError("document missing end-of-diagnostics")
        if line.startswith("case "):
            if current_case is not None:
                raise DocumentError("case metadata is not followed by a geometry")
            current_case = read_case()
            cases.append(current_case)
            continue
        if line == "geometry" or line.startswith("geometry "):
            if current_case is None:
                raise DocumentError("geometry is missing its case metadata")
            geometries.append(read_geometry(current_case))
            current_case = None
            continue
        if line.startswith("end-of-diagnostics"):
            terminal_line = parser.next()
            break
        raise DocumentError(f"unexpected token {line!r} at top level")
    if current_case is not None:
        raise DocumentError("terminal case metadata is missing a geometry")
    return {"case": cases, "geometries": geometries, "terminal_line": terminal_line}


def occupied_mask(occupations: FloatArray) -> NDArray[np.bool_]:
    """Select restricted orbitals with total occupation above one half."""
    return occupations[0] + occupations[1] > 0.5


def _principal_metrics(cross: FloatArray, target_dimension: int) -> Record:
    """Convert a cross-subspace Gram matrix into angle and capture metrics."""
    singular_values = np.linalg.svd(cross, compute_uv=False)
    if not np.all(np.isfinite(singular_values)):
        raise DocumentError("subspace singular values are non-finite")
    largest = float(np.max(singular_values, initial=0.0))
    if largest > 1.0 + _SINGULAR_VALUE_TOLERANCE:
        raise DocumentError(
            f"subspace singular value {largest:.6g} exceeds the physical bound"
        )
    singular_values = np.clip(singular_values, 0.0, 1.0)
    if singular_values.size == 0 or target_dimension == 0:
        raise DocumentError("occupied subspace is empty")
    cos_min = float(np.min(singular_values))
    cos_squared = singular_values * singular_values
    return {
        "subspace_min_cos": cos_min,
        "subspace_max_angle_deg": math.degrees(math.acos(cos_min)),
        "subspace_capture_fraction": float(np.sum(cos_squared) / target_dimension),
        "subspace_chordal_distance": math.sqrt(float(np.sum(1.0 - cos_squared))),
    }


def subspace_metrics(
    previous_coefficients: FloatArray,
    current_coefficients: FloatArray,
    previous_occupations: FloatArray,
    current_occupations: FloatArray,
    overlap: FloatArray,
) -> Record:
    """Measure two occupied subspaces expressed in one AO basis."""
    previous = previous_coefficients[0][:, occupied_mask(previous_occupations)]
    current = current_coefficients[0][:, occupied_mask(current_occupations)]
    return _principal_metrics(previous.T @ overlap @ current, current.shape[1])


def cross_geometry_subspace_metrics(source: Record, target: Record) -> Record:
    """Measure physical occupied-space angles with the cross-AO overlap."""
    cross_overlap = target["cross_overlap_from_source"]
    if cross_overlap is None:
        raise DocumentError("trajectory target is missing its cross-AO overlap")
    source_state = source["converged"]
    target_state = target["converged"]
    source_coefficients = source_state["C"][0][:, occupied_mask(source_state["occ"])]
    target_coefficients = target_state["C"][0][:, occupied_mask(target_state["occ"])]
    cross = source_coefficients.T @ cross_overlap @ target_coefficients
    return _principal_metrics(cross, target_coefficients.shape[1])


def density_overlap(first: FloatArray, second: FloatArray) -> float:
    """Return normalized Frobenius overlap for one common AO representation."""
    denominator = math.sqrt(
        float(np.trace(first @ first)) * float(np.trace(second @ second))
    )
    if denominator <= 0.0:
        raise DocumentError("density norm is not positive")
    return float(np.trace(first @ second)) / denominator


def cross_geometry_density_metrics(source: Record, target: Record) -> Record:
    """Compare density operators using ``S1``, ``S2``, and physical ``S12``."""
    source_density = source["converged"]["P"]
    target_density = target["converged"]["P"]
    source_overlap = source["overlap"]
    target_overlap = target["overlap"]
    cross_overlap = target["cross_overlap_from_source"]
    if cross_overlap is None:
        raise DocumentError("trajectory target is missing its cross-AO overlap")
    reverse_overlap = cross_overlap.T
    source_norm_squared = float(
        np.trace(source_density @ source_overlap @ source_density @ source_overlap)
    )
    target_norm_squared = float(
        np.trace(target_density @ target_overlap @ target_density @ target_overlap)
    )
    inner = float(
        np.trace(source_density @ cross_overlap @ target_density @ reverse_overlap)
    )
    if source_norm_squared <= 0.0 or target_norm_squared <= 0.0:
        raise DocumentError("cross-geometry density norm is not positive")
    distance_squared = max(source_norm_squared + target_norm_squared - 2.0 * inner, 0.0)
    normalized_inner = inner / math.sqrt(source_norm_squared * target_norm_squared)
    return {
        "rel_density_operator_change": math.sqrt(distance_squared)
        / math.sqrt(source_norm_squared),
        "density_operator_overlap": min(max(normalized_inner, -1.0), 1.0),
    }


def generalized_residual(
    new_hamiltonian: FloatArray,
    old_coefficients: FloatArray,
    old_eigenvalues: FloatArray,
    overlap: FloatArray,
    old_occupations: FloatArray,
) -> tuple[float, float]:
    """Measure full and occupied residuals against a new generalized problem."""
    coefficients = old_coefficients[0]
    eigenvalues = old_eigenvalues[0]
    hamiltonian_norm = max(float(np.linalg.norm(new_hamiltonian)), 1.0e-300)
    full = new_hamiltonian @ coefficients - overlap @ coefficients @ np.diag(
        eigenvalues
    )
    mask = occupied_mask(old_occupations)
    occupied_coefficients = coefficients[:, mask]
    if occupied_coefficients.shape[1] == 0:
        raise DocumentError("occupied residual has no occupied columns")
    occupied = (
        new_hamiltonian @ occupied_coefficients
        - overlap @ occupied_coefficients @ np.diag(eigenvalues[mask])
    )
    return (
        float(np.linalg.norm(full)) / hamiltonian_norm,
        float(np.linalg.norm(occupied)) / hamiltonian_norm,
    )


def rr_eigenvalue_error(
    new_hamiltonian: FloatArray,
    previous_coefficients: FloatArray,
    current_eigenvalues: FloatArray,
    previous_occupations: FloatArray,
    current_occupations: FloatArray,
) -> tuple[float, float]:
    """Compare occupied Rayleigh--Ritz and target eigenvalues."""
    coefficients = previous_coefficients[0][:, occupied_mask(previous_occupations)]
    target_eigenvalues = current_eigenvalues[0][occupied_mask(current_occupations)]
    if coefficients.shape[1] == 0 or target_eigenvalues.size == 0:
        raise DocumentError("Rayleigh--Ritz comparison has no occupied orbitals")
    projected = coefficients.T @ new_hamiltonian @ coefficients
    eigenvalues = np.linalg.eigvalsh((projected + projected.T) / 2.0)
    compared = min(eigenvalues.size, target_eigenvalues.size)
    difference = np.abs(
        np.sort(eigenvalues)[:compared] - np.sort(target_eigenvalues)[:compared]
    )
    return (
        float(np.max(difference)),
        math.sqrt(float(np.mean(difference * difference))),
    )


def validate_eigenpairs(
    hamiltonian: FloatArray,
    coefficients: FloatArray,
    eigenvalues: FloatArray,
    overlap: FloatArray,
) -> tuple[float, float]:
    """Validate restricted generalized eigenpairs and metric orthonormality."""
    coefficient_matrix = coefficients[0]
    eigenvalue_vector = eigenvalues[0]
    metric_error = coefficient_matrix.T @ overlap @ coefficient_matrix - np.eye(
        coefficient_matrix.shape[0]
    )
    residual = (
        hamiltonian @ coefficient_matrix
        - overlap @ coefficient_matrix @ np.diag(eigenvalue_vector)
    )
    hamiltonian_norm = float(np.linalg.norm(hamiltonian)) or 1.0
    return (
        float(np.max(np.abs(metric_error))),
        float(np.max(np.abs(residual))) / hamiltonian_norm,
    )


def analyze_geometry(geometry: Record) -> Record:
    """Compute all same-geometry per-iteration reuse diagnostics."""
    overlap = geometry["overlap"]
    rows: list[Record] = []
    for index, iteration in enumerate(geometry["iterations"]):
        row: Record = {
            "k": iteration["k"],
            "step_micros": iteration["step_micros"],
            "eigensolve_micros": iteration["eigensolve_micros"],
        }
        metric_error, residual_error = validate_eigenpairs(
            iteration["H"], iteration["C"], iteration["eps"], overlap
        )
        row["validation_ctsc_max"] = metric_error
        row["validation_he_max"] = residual_error
        if index > 0:
            previous = geometry["iterations"][index - 1]
            hamiltonian_change = float(np.linalg.norm(iteration["H"] - previous["H"]))
            row["rel_dH"] = hamiltonian_change / (
                float(np.linalg.norm(previous["H"])) or 1.0
            )
            density_change = float(np.linalg.norm(iteration["P"] - previous["P"]))
            row["rel_dP"] = density_change / (
                float(np.linalg.norm(previous["P"])) or 1.0
            )
            row["density_overlap"] = density_overlap(previous["P"], iteration["P"])
            row.update(
                subspace_metrics(
                    previous["C"],
                    iteration["C"],
                    previous["occ"],
                    iteration["occ"],
                    overlap,
                )
            )
            full_residual, occupied_residual = generalized_residual(
                iteration["H"],
                previous["C"],
                previous["eps"],
                overlap,
                previous["occ"],
            )
            row["rel_residual_full"] = full_residual
            row["rel_residual_occupied"] = occupied_residual
            maximum_error, rms_error = rr_eigenvalue_error(
                iteration["H"],
                previous["C"],
                iteration["eps"],
                previous["occ"],
                iteration["occ"],
            )
            row["rr_eigenvalue_max_err"] = maximum_error
            row["rr_eigenvalue_rms_err"] = rms_error
        rows.append(row)

    terminal = geometry["converged"]
    terminal_metric_error, terminal_residual_error = validate_eigenpairs(
        terminal["H"], terminal["C"], terminal["eps"], overlap
    )
    block: Record = {
        "generation": geometry["generation"],
        "role": geometry["role"],
        "start_mode": geometry["start_mode"],
        "nao": geometry["nao"],
        "iterations": rows,
        "converged_state": geometry["converged_state"],
        "terminal_validation_ctsc_max": terminal_metric_error,
        "terminal_validation_he_max": terminal_residual_error,
    }
    if rows:
        block["eigensolve_total_micros"] = sum(row["eigensolve_micros"] for row in rows)
        block["step_total_micros"] = sum(row["step_micros"] for row in rows)
        block["eigensolve_share"] = block["eigensolve_total_micros"] / max(
            block["step_total_micros"], 1
        )
    return block


def target_metric_reuse_metrics(source: Record, target: Record) -> Record:
    """Test a co-moving AO-label candidate against the target SCC problem.

    This is an algorithmic transport, not a physical cross-AO overlap. Source
    occupied coefficients keep their atom/AO labels, are reorthonormalized in
    ``S2``, and receive a Rayleigh--Ritz rotation with the target terminal
    effective Hamiltonian before residuals are measured.
    """
    source_state = source["converged"]
    target_state = target["converged"]
    source_occupied = source_state["C"][0][:, occupied_mask(source_state["occ"])]
    target_occupied = target_state["C"][0][:, occupied_mask(target_state["occ"])]
    if source_occupied.shape[1] != target_occupied.shape[1]:
        raise DocumentError("trajectory changes the occupied-space dimension")
    target_overlap = target["overlap"]
    gram = source_occupied.T @ target_overlap @ source_occupied
    gram_values, gram_vectors = np.linalg.eigh((gram + gram.T) / 2.0)
    minimum_gram = float(np.min(gram_values))
    if minimum_gram <= _GRAM_EIGENVALUE_TOLERANCE:
        raise DocumentError(
            f"co-moving AO-label transport is ill-conditioned ({minimum_gram:.3e})"
        )
    inverse_root = gram_vectors @ np.diag(1.0 / np.sqrt(gram_values)) @ gram_vectors.T
    orthonormal_candidate = source_occupied @ inverse_root
    target_hamiltonian = target_state["H"]
    projected = orthonormal_candidate.T @ target_hamiltonian @ orthonormal_candidate
    projected_values, rotation = np.linalg.eigh((projected + projected.T) / 2.0)
    rotated_candidate = orthonormal_candidate @ rotation
    residual = (
        target_hamiltonian @ rotated_candidate
        - target_overlap @ rotated_candidate @ np.diag(projected_values)
    )
    residual_relative = float(np.linalg.norm(residual)) / max(
        float(np.linalg.norm(target_hamiltonian)), 1.0e-300
    )
    target_values = target_state["eps"][0][occupied_mask(target_state["occ"])]
    difference = np.abs(np.sort(projected_values) - np.sort(target_values))
    metrics = _principal_metrics(
        rotated_candidate.T @ target_overlap @ target_occupied,
        target_occupied.shape[1],
    )
    return {
        "transport": "co_moving_ao_labels_reorthonormalized_in_target_metric",
        "gram_min_eigenvalue": minimum_gram,
        "rr_rel_residual_occupied": residual_relative,
        "rr_eigenvalue_max_err": float(np.max(difference)),
        "rr_eigenvalue_rms_err": math.sqrt(float(np.mean(difference * difference))),
        "rr_target_subspace_capture_fraction": metrics["subspace_capture_fraction"],
        "rr_target_subspace_max_angle_deg": metrics["subspace_max_angle_deg"],
    }


def trajectory_metrics(source: Record, target: Record) -> Record:
    """Measure physical and algorithmic source-to-target reuse quality."""
    if not source["converged_state"] or not target["converged_state"]:
        raise DocumentError("trajectory metrics require two converged endpoints")
    result: Record = {}
    result.update(cross_geometry_subspace_metrics(source, target))
    result.update(cross_geometry_density_metrics(source, target))
    result["target_metric_reuse"] = target_metric_reuse_metrics(source, target)
    return result


def _same_policy(first: Record, second: Record) -> bool:
    """Return whether two case metadata blocks share strict WARM identity."""
    keys = (
        "nat",
        "atomic_numbers",
        "molecular_charge",
        "unpaired_electrons",
        "temperature_kelvin",
        "mixer_memory",
        "mixer_damping",
        "maximum_iterations",
    )
    return all(first[key] == second[key] for key in keys)


def warm_start_control(warm: Record, fresh: Record) -> Record:
    """Compare WARM and FRESH runs of the exact same target geometry."""
    if warm["case"]["positions"] != fresh["case"]["positions"]:
        raise DocumentError("WARM/FRESH control geometries differ")
    if not _same_policy(warm["case"], fresh["case"]):
        raise DocumentError("WARM/FRESH control SCC policies differ")
    if not warm["converged_state"] or not fresh["converged_state"]:
        raise DocumentError("WARM/FRESH control requires two converged runs")
    warm_iterations = len(warm["iterations"])
    fresh_iterations = len(fresh["iterations"])
    final_metrics = subspace_metrics(
        warm["converged"]["C"],
        fresh["converged"]["C"],
        warm["converged"]["occ"],
        fresh["converged"]["occ"],
        warm["overlap"],
    )
    density_difference = float(
        np.linalg.norm(warm["converged"]["P"] - fresh["converged"]["P"])
    ) / (float(np.linalg.norm(fresh["converged"]["P"])) or 1.0)
    return {
        "warm_iterations": warm_iterations,
        "fresh_iterations": fresh_iterations,
        "iteration_reduction": fresh_iterations - warm_iterations,
        "final_density_rel_difference": density_difference,
        "final_subspace_capture_fraction": final_metrics["subspace_capture_fraction"],
        "final_subspace_max_angle_deg": final_metrics["subspace_max_angle_deg"],
    }


def build_report(path: Path, document: Record) -> Record:
    """Build a compact JSON-serializable report from one parsed stream."""
    geometries = [analyze_geometry(geometry) for geometry in document["geometries"]]
    report: Record = {
        "document": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case": document["case"],
        "geometries": geometries,
    }
    roles = {geometry["role"]: geometry for geometry in document["geometries"]}
    trajectory_roles = {"source", "target_warm", "target_fresh"}
    if trajectory_roles.issubset(roles):
        report["trajectory"] = trajectory_metrics(roles["source"], roles["target_warm"])
        report["warm_start_control"] = warm_start_control(
            roles["target_warm"], roles["target_fresh"]
        )
    elif len(document["geometries"]) != 1:
        raise DocumentError("trajectory document is missing a required run role")
    return report


def summarize(report: Record) -> str:
    """Render a concise human-readable summary."""
    lines: list[str] = []
    for geometry in report["geometries"]:
        rows = geometry["iterations"]
        convergence = "converged" if geometry["converged_state"] else "NOT converged"
        lines.append(
            f"{geometry['role']} ({geometry['start_mode']}): "
            f"nao={geometry['nao']} iterations={len(rows)} [{convergence}]"
        )
        if rows:
            lines.append(
                f"  eigensolve total {geometry['eigensolve_total_micros']} us, "
                f"share of step {geometry['eigensolve_share']:.2f}"
            )
        if len(rows) >= 2:
            lines.append(
                "  ang_max(deg)   "
                + " ".join(f"{row['subspace_max_angle_deg']:9.3f}" for row in rows[1:])
            )
            lines.append(
                "  capture frac   "
                + " ".join(
                    f"{row['subspace_capture_fraction']:9.4f}" for row in rows[1:]
                )
            )
            lines.append(
                "  RR eig err     "
                + " ".join(f"{row['rr_eigenvalue_max_err']:9.3e}" for row in rows[1:])
            )
    if "trajectory" in report:
        trajectory = report["trajectory"]
        transport = trajectory["target_metric_reuse"]
        lines.append(
            "trajectory physical: "
            f"capture={trajectory['subspace_capture_fraction']:.4f} "
            f"angle={trajectory['subspace_max_angle_deg']:.3f} deg"
        )
        lines.append(
            "trajectory target-metric RR: "
            f"residual={transport['rr_rel_residual_occupied']:.3e} "
            f"eig_err={transport['rr_eigenvalue_max_err']:.3e}"
        )
        control = report["warm_start_control"]
        lines.append(
            "target control: "
            f"warm={control['warm_iterations']} fresh={control['fresh_iterations']}"
        )
    return "\n".join(lines)


def main(arguments: list[str] | None = None) -> int:
    """Analyze documents and optionally write one compact JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "documents", nargs="+", type=Path, help="capture diagnostic stream(s)"
    )
    parser.add_argument("--report", type=Path, help="write JSON report to this path")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress console summaries"
    )
    options = parser.parse_args(arguments)

    reports: list[Record] = []
    for path in options.documents:
        report = build_report(path, parse_document(path))
        reports.append(report)
        if not options.quiet:
            sys.stdout.write(summarize(report) + "\n\n")
    if options.report is not None:
        with options.report.open("w", encoding="utf-8") as handle:
            json.dump(
                reports if len(reports) > 1 else reports[0],
                handle,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
