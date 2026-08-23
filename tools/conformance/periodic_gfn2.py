#!/usr/bin/env python3
"""Generate and verify independent native-3D-periodic GFN2 evidence.

Neutral full-model values come from a hash-pinned tblite executable. Charged
uniform-background values are reconstructed analytically because the reviewed
tblite 0.7.0 monopole matrix does not contain the background constant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/conformance/periodic/manifest.json"
ACCURACY = 1.0e-4
ACCURACY_TEXT = "0.0001"
SCHEMA = "xtbloom-periodic-gfn2-conformance-v1"
GOLDEN_SCHEMA = "xtbloom-periodic-gfn2-golden-v1"
STRAIN_STEPS = (2.0e-4, 1.0e-4, 5.0e-5)
STRAIN_MODES = (
    ("xx", 0, 0),
    ("yy", 1, 1),
    ("zz", 2, 2),
    ("xy", 0, 1),
    ("xz", 0, 2),
    ("yz", 1, 2),
)


class PeriodicOracleError(RuntimeError):
    """An actionable periodic-corpus or oracle failure."""


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and retain the path in malformed-input errors."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PeriodicOracleError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PeriodicOracleError(f"expected a JSON object in {path}")
    return value


def dump_json(path: Path, value: dict[str, Any]) -> None:
    """Write the canonical, review-friendly representation of generated data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without assuming it fits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one value using the corpus canonical JSON serialization."""
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def repository_path(relative: str) -> Path:
    """Resolve a manifest path while forbidding escape from the repository."""
    candidate = (REPOSITORY_ROOT / relative).resolve()
    try:
        candidate.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise PeriodicOracleError(f"manifest path escapes repository: {relative}") from exc
    return candidate


def finite_number(value: object, label: str) -> float:
    """Return a finite binary64-compatible numeric value."""
    if type(value) not in (int, float):
        raise PeriodicOracleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PeriodicOracleError(f"{label} must be finite")
    return result


def finite_text_number(value: str, label: str) -> float:
    """Parse one finite number from the textual Turbomole grammar."""
    try:
        result = float(value)
    except ValueError as exc:
        raise PeriodicOracleError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise PeriodicOracleError(f"{label} must be finite")
    return result


def finite_vector(value: object, count: int, label: str) -> list[float]:
    """Validate a fixed-length numeric vector."""
    if not isinstance(value, list) or len(value) != count:
        raise PeriodicOracleError(f"{label} must contain {count} values")
    return [finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def determinant(cell: list[float]) -> float:
    """Return the determinant of a row-major 3-by-3 matrix."""
    return (
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6])
    )


def parse_turbomole(path: Path) -> dict[str, Any]:
    """Parse the deliberately narrow periodic Turbomole fixture grammar."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    try:
        coord_begin = lines.index("$coord") + 1
        periodic_line = lines.index("$periodic 3")
        lattice_begin = lines.index("$lattice") + 1
        end_line = lines.index("$end")
    except ValueError as exc:
        raise PeriodicOracleError(
            f"{path} must contain $coord, $periodic 3, $lattice, and $end"
        ) from exc
    if not (coord_begin <= periodic_line < lattice_begin <= end_line):
        raise PeriodicOracleError(f"{path} has malformed section ordering")
    coord_lines = [line for line in lines[coord_begin:periodic_line] if line]
    lattice_lines = [line for line in lines[lattice_begin:end_line] if line]
    if not coord_lines or len(lattice_lines) != 3:
        raise PeriodicOracleError(f"{path} must contain atoms and three lattice rows")

    positions: list[float] = []
    symbols: list[str] = []
    for index, line in enumerate(coord_lines):
        fields = line.split()
        if len(fields) != 4 or not re.fullmatch(r"[A-Za-z]{1,2}", fields[3]):
            raise PeriodicOracleError(f"{path} coordinate row {index} is malformed")
        positions.extend(
            finite_text_number(value, f"{path} coordinate") for value in fields[:3]
        )
        symbols.append(fields[3][0].upper() + fields[3][1:].lower())

    cell: list[float] = []
    for index, line in enumerate(lattice_lines):
        fields = line.split()
        if len(fields) != 3:
            raise PeriodicOracleError(f"{path} lattice row {index} is malformed")
        cell.extend(finite_text_number(value, f"{path} lattice") for value in fields)
    volume = determinant(cell)
    if not (volume > 0.0) or not math.isfinite(volume):
        raise PeriodicOracleError(f"{path} cell must be finite and right-handed")
    return {
        "cell_matrix_row_major_bohr": cell,
        "positions_bohr": positions,
        "symbols": symbols,
        "volume_bohr3": volume,
    }


def format_turbomole(
    structure: dict[str, Any], positions: list[float], cell: list[float]
) -> str:
    """Materialize one affine-deformed periodic fixture for finite differences."""
    lines = ["$coord"]
    for atom, symbol in enumerate(structure["symbols"]):
        xyz = positions[3 * atom : 3 * atom + 3]
        lines.append(
            f"  {xyz[0]:.17g}  {xyz[1]:.17g}  {xyz[2]:.17g}  {symbol}"
        )
    lines.extend(["$periodic 3", "$lattice"])
    for row in range(3):
        values = cell[3 * row : 3 * row + 3]
        lines.append(f"  {values[0]:.17g}  {values[1]:.17g}  {values[2]:.17g}")
    lines.extend(["$end", ""])
    return "\n".join(lines)


def affine_deformation(
    structure: dict[str, Any], row: int, column: int, step: float
) -> tuple[list[float], list[float]]:
    """Apply r'=(I+eps)r and H'=H(I+eps)^T in row-major storage."""
    deformation = [0.0] * 9
    deformation[0] = deformation[4] = deformation[8] = 1.0
    deformation[3 * row + column] += step

    positions = structure["positions_bohr"]
    deformed_positions: list[float] = []
    for atom in range(len(positions) // 3):
        vector = positions[3 * atom : 3 * atom + 3]
        deformed_positions.extend(
            sum(deformation[3 * component + source] * vector[source] for source in range(3))
            for component in range(3)
        )

    cell = structure["cell_matrix_row_major_bohr"]
    deformed_cell = [0.0] * 9
    for lattice_row in range(3):
        for cartesian_column in range(3):
            deformed_cell[3 * lattice_row + cartesian_column] = sum(
                cell[3 * lattice_row + source]
                * deformation[3 * cartesian_column + source]
                for source in range(3)
            )
    return deformed_positions, deformed_cell


def command_template() -> list[str]:
    """Return the path-independent reviewed tblite CLI command."""
    return [
        "{executable}",
        "run",
        "--method",
        "gfn2",
        "--acc",
        ACCURACY_TEXT,
        "--no-restart",
        "--grad",
        "{gradient}",
        "--json",
        "{json}",
        "[--charge {charge}]",
        "[--spin {unpaired_electrons}]",
        "{input}",
    ]


def tblite_command(
    executable: Path, case: dict[str, Any], input_path: Path, output_path: Path
) -> list[str]:
    """Build the exact single-point invocation for one periodic case."""
    command = [
        str(executable),
        "run",
        "--method",
        "gfn2",
        "--acc",
        ACCURACY_TEXT,
        "--no-restart",
        "--grad",
        str(output_path.with_suffix(".gradient")),
        "--json",
        str(output_path),
    ]
    charge = int(case["molecular_charge"])
    if charge:
        command.extend(["--charge", str(charge)])
    unpaired = int(case["unpaired_electrons"])
    if unpaired:
        command.extend(["--spin", str(unpaired)])
    command.append(str(input_path))
    return command


def oracle_environment() -> tuple[dict[str, str], dict[str, Any]]:
    """Force the deterministic thread and restart environment boundary."""
    environment = os.environ.copy()
    removed = sorted(name for name in environment if name.upper().startswith("XTB"))
    for name in removed:
        environment.pop(name, None)
    fixed = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    environment.update(fixed)
    return environment, {
        "inherited_environment_boundary": (
            "All non-XTB variables except the recorded thread controls are inherited; "
            "runtime identity is pinned independently."
        ),
        "removed_variables": removed,
        "set": fixed,
    }


def executable_version(executable: Path, environment: dict[str, str]) -> str:
    """Read the exact CLI version text."""
    completed = subprocess.run(
        [str(executable), "--version"],
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise PeriodicOracleError(
            f"cannot query tblite version ({completed.returncode}):\n{completed.stdout}"
        )
    return completed.stdout.strip()


def discover_libtblite(
    executable: Path, environment: dict[str, str], expected_sha256: str
) -> dict[str, str]:
    """Resolve and hash the libtblite selected by the executable loader."""
    completed = subprocess.run(
        ["ldd", str(executable)],
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise PeriodicOracleError(f"ldd failed for {executable}:\n{completed.stdout}")
    match = re.search(r"\blibtblite\.so\S*\s+=>\s+(\S+)", completed.stdout)
    if match is None:
        raise PeriodicOracleError("cannot resolve libtblite through ldd")
    library = Path(match.group(1)).resolve()
    digest = sha256_file(library)
    if digest != expected_sha256:
        raise PeriodicOracleError(
            f"libtblite SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return {
        "discovery": "ldd",
        "filename": library.name,
        "sha256": digest,
        "status": "resolved",
    }


def normalize_tblite(raw: dict[str, Any], atom_count: int) -> dict[str, Any]:
    """Normalize tblite JSON into xTBloom units and row-major strain order."""
    energy = finite_number(raw.get("energy"), "tblite energy")
    gradient = finite_vector(raw.get("gradient"), 3 * atom_count, "tblite gradient")
    # tblite reshapes its Fortran [3,3] array in column-major order.
    virial_column_major = finite_vector(raw.get("virial"), 9, "tblite virial")
    strain_row_major = [
        virial_column_major[3 * column + row]
        for row in range(3)
        for column in range(3)
    ]
    return {
        "energy_hartree": energy,
        "forces_hartree_per_bohr": [-value for value in gradient],
        "gradient_hartree_per_bohr": gradient,
        "strain_derivatives_hartree": strain_row_major,
    }


def run_tblite(
    executable: Path,
    case: dict[str, Any],
    input_path: Path,
    output_path: Path,
    environment: dict[str, str],
) -> tuple[dict[str, Any], str]:
    """Execute one isolated oracle calculation and normalize its output."""
    command = tblite_command(executable, case, input_path, output_path)
    completed = subprocess.run(
        command,
        cwd=output_path.parent,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise PeriodicOracleError(
            f"tblite failed for {case['id']} ({completed.returncode}):\n{completed.stdout}"
        )
    raw = load_json(output_path)
    return normalize_tblite(raw, int(case["atom_count"])), sha256_file(output_path)


def strain_finite_differences(
    executable: Path,
    case: dict[str, Any],
    structure: dict[str, Any],
    work: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Generate three-step central differences for six ordered strain modes."""
    evidence: dict[str, Any] = {
        "affine_convention": "r'=(I+epsilon)r; H'=H(I+epsilon)^T",
        "modes": {},
        "steps": list(STRAIN_STEPS),
    }
    for mode, row, column in STRAIN_MODES:
        estimates: list[float] = []
        for step_index, step in enumerate(STRAIN_STEPS):
            energies: dict[int, float] = {}
            for sign in (-1, 1):
                positions, cell = affine_deformation(
                    structure, row, column, sign * step
                )
                input_path = work / f"{case['id']}-{mode}-{step_index}-{sign:+d}.tmol"
                input_path.write_text(
                    format_turbomole(structure, positions, cell), encoding="utf-8"
                )
                output_path = input_path.with_suffix(".json")
                properties, _ = run_tblite(
                    executable, case, input_path, output_path, environment
                )
                energies[sign] = float(properties["energy_hartree"])
            estimates.append((energies[1] - energies[-1]) / (2.0 * step))
        # The two finest estimates give the ordinary O(h^2) Richardson value.
        richardson = (4.0 * estimates[-1] - estimates[-2]) / 3.0
        evidence["modes"][mode] = {
            "component": [row, column],
            "central_differences_hartree": estimates,
            "richardson_hartree": richardson,
        }
    return evidence


def selected_cases(manifest: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    """Select unique case IDs and reject command-line typos."""
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise PeriodicOracleError("manifest cases must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise PeriodicOracleError("each case must have a string ID")
        if case["id"] in by_id:
            raise PeriodicOracleError(f"duplicate case ID: {case['id']}")
        by_id[case["id"]] = case
    if not names:
        return list(by_id.values())
    unknown = sorted(set(names) - set(by_id))
    if unknown:
        raise PeriodicOracleError(f"unknown case ID(s): {', '.join(unknown)}")
    return [by_id[name] for name in names]


def generate(
    manifest_path: Path, executable_arg: Path, output_dir: Path, names: list[str] | None
) -> None:
    """Generate neutral primary and charged diagnostic tblite results."""
    manifest = load_json(manifest_path)
    reference = manifest["reference_engine"]
    executable = Path(shutil.which(str(executable_arg)) or executable_arg).resolve()
    if not executable.is_file():
        raise PeriodicOracleError(f"tblite executable does not exist: {executable_arg}")
    if sha256_file(executable) != reference["executable_sha256"]:
        raise PeriodicOracleError("tblite executable SHA-256 does not match manifest")
    environment, environment_provenance = oracle_environment()
    version = executable_version(executable, environment)
    if f"tblite version {reference['version']}" not in version:
        raise PeriodicOracleError(f"unexpected tblite version:\n{version}")
    runtime = {
        "libtblite": discover_libtblite(
            executable, environment, reference["runtime_artifacts"]["libtblite_sha256"]
        )
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="xtbloom-periodic-oracle-") as temporary:
        work = Path(temporary)
        for case in selected_cases(manifest, names):
            if case["oracle_role"] == "analytic-background":
                continue
            input_path = repository_path(case["input"])
            raw_path = work / f"{case['id']}.json"
            properties, raw_digest = run_tblite(
                executable, case, input_path, raw_path, environment
            )
            structure = parse_turbomole(input_path)
            finite_difference = None
            if case.get("strain_finite_difference", False):
                finite_difference = strain_finite_differences(
                    executable, case, structure, work, environment
                )
            provenance = {
                "accuracy": ACCURACY,
                "command_template": command_template(),
                "engine": "tblite",
                "environment": environment_provenance,
                "executable_sha256": sha256_file(executable),
                "executable_version": version,
                "generation_mode": "live-cli",
                "input": case["input"],
                "input_sha256": sha256_file(input_path),
                "raw_output_sha256": raw_digest,
                "runtime": runtime,
                "source_revision": reference["revision"],
            }
            golden: dict[str, Any] = {
                "case_id": case["id"],
                "oracle_role": case["oracle_role"],
                "properties": properties,
                "provenance": provenance,
                "schema": GOLDEN_SCHEMA,
                "source_output_sha256": sha256_json(properties),
            }
            if finite_difference is not None:
                golden["strain_finite_difference"] = finite_difference
            destination = output_dir / f"{case['id']}.json"
            dump_json(destination, golden)
            print(f"generated {destination}")  # noqa: T201 - CLI progress


def check_source_modules(reference: dict[str, Any]) -> None:
    """Validate immutable source identifiers without requiring a checkout."""
    modules = reference.get("source_modules")
    if not isinstance(modules, dict) or not modules:
        raise PeriodicOracleError("reference source_modules must be a nonempty object")
    for path, identity in modules.items():
        if not isinstance(path, str) or not isinstance(identity, dict):
            raise PeriodicOracleError("source module entries are malformed")
        if re.fullmatch(r"[0-9a-f]{40}", str(identity.get("git_blob", ""))) is None:
            raise PeriodicOracleError(f"{path} has an invalid Git blob ID")
        if re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256", ""))) is None:
            raise PeriodicOracleError(f"{path} has an invalid SHA-256")
        if not isinstance(identity.get("role"), str) or not identity["role"]:
            raise PeriodicOracleError(f"{path} has no reviewed role")


def check_strain_evidence(golden: dict[str, Any], strain: list[float]) -> None:
    """Require convergent multi-step evidence for all six released modes."""
    evidence = golden.get("strain_finite_difference")
    if not isinstance(evidence, dict) or evidence.get("steps") != list(STRAIN_STEPS):
        raise PeriodicOracleError(f"{golden['case_id']} lacks canonical strain steps")
    modes = evidence.get("modes")
    if not isinstance(modes, dict) or set(modes) != {mode[0] for mode in STRAIN_MODES}:
        raise PeriodicOracleError(f"{golden['case_id']} lacks all six strain modes")
    for name, row, column in STRAIN_MODES:
        mode = modes[name]
        if mode.get("component") != [row, column]:
            raise PeriodicOracleError(f"{golden['case_id']} {name} component drifted")
        estimates = finite_vector(
            mode.get("central_differences_hartree"),
            len(STRAIN_STEPS),
            f"{golden['case_id']} {name} strain differences",
        )
        richardson = finite_number(
            mode.get("richardson_hartree"), f"{golden['case_id']} {name} Richardson"
        )
        expected_richardson = (4.0 * estimates[-1] - estimates[-2]) / 3.0
        if richardson != expected_richardson:
            raise PeriodicOracleError(f"{golden['case_id']} {name} Richardson drifted")
        analytic = strain[3 * row + column]
        if abs(richardson - analytic) > 2.0e-5:
            raise PeriodicOracleError(
                f"{golden['case_id']} {name} strain mismatch: "
                f"analytic {analytic}, finite difference {richardson}"
            )


def check_analytic_background(manifest: dict[str, Any]) -> None:
    """Reconstruct the charged-cell background energy, potential, and strain."""
    cases = manifest.get("analytic_background_cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise PeriodicOracleError("at least two analytic background cases are required")
    for case in cases:
        charge = finite_number(case.get("charge_e"), "background charge")
        alpha = finite_number(case.get("alpha_bohr_inverse"), "background alpha")
        volume = finite_number(case.get("volume_bohr3"), "background volume")
        if not alpha > 0.0 or not volume > 0.0 or charge == 0.0:
            raise PeriodicOracleError("background cases require Q!=0, alpha>0, V>0")
        factor = math.pi / (alpha * alpha * volume)
        expected_energy = -0.5 * factor * charge * charge
        expected_potential = -factor * charge
        expected_strain = [0.0] * 9
        for diagonal in (0, 4, 8):
            expected_strain[diagonal] = -expected_energy
        if finite_number(case.get("energy_hartree"), "background energy") != expected_energy:
            raise PeriodicOracleError(f"{case.get('id')} background energy drifted")
        if (
            finite_number(case.get("potential_hartree_per_e"), "background potential")
            != expected_potential
        ):
            raise PeriodicOracleError(f"{case.get('id')} background potential drifted")
        if finite_vector(case.get("strain_derivatives_hartree"), 9, "background strain") != expected_strain:
            raise PeriodicOracleError(f"{case.get('id')} background strain drifted")


def check(manifest_path: Path) -> None:
    """Verify all committed periodic evidence without running tblite."""
    manifest = load_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise PeriodicOracleError(f"manifest schema must be {SCHEMA}")
    if manifest.get("units") != {
        "cell": "bohr",
        "energy": "hartree",
        "forces": "hartree_per_bohr",
        "positions": "bohr",
        "strain_derivative": "hartree",
    }:
        raise PeriodicOracleError("manifest atomic-unit contract drifted")
    reference = manifest.get("reference_engine")
    if not isinstance(reference, dict):
        raise PeriodicOracleError("manifest reference_engine must be an object")
    if reference.get("revision") != "133f91efb94b47f05848e1f86832f40a1accc385":
        raise PeriodicOracleError("tblite source revision drifted")
    if reference.get("version") != "0.7.0" or reference.get("accuracy") != ACCURACY:
        raise PeriodicOracleError("tblite version or accuracy drifted")
    if reference.get("cli_command_template") != command_template():
        raise PeriodicOracleError("tblite command template drifted")
    for name in ("executable_sha256",):
        if re.fullmatch(r"[0-9a-f]{64}", str(reference.get(name, ""))) is None:
            raise PeriodicOracleError(f"reference {name} is invalid")
    runtime = reference.get("runtime_artifacts")
    if not isinstance(runtime, dict) or re.fullmatch(
        r"[0-9a-f]{64}", str(runtime.get("libtblite_sha256", ""))
    ) is None:
        raise PeriodicOracleError("reference libtblite hash is invalid")
    check_source_modules(reference)

    primary_count = 0
    diagnostic_count = 0
    for case in selected_cases(manifest, None):
        if case.get("periodic_axes") != 7:
            raise PeriodicOracleError(f"{case['id']} must use XYZ periodicity")
        input_path = repository_path(case["input"])
        if sha256_file(input_path) != case.get("input_sha256"):
            raise PeriodicOracleError(f"{case['id']} input SHA-256 mismatch")
        structure = parse_turbomole(input_path)
        if len(structure["symbols"]) != int(case["atom_count"]):
            raise PeriodicOracleError(f"{case['id']} atom count mismatch")
        if structure["cell_matrix_row_major_bohr"] != finite_vector(
            case.get("cell_matrix_row_major_bohr"), 9, f"{case['id']} cell"
        ):
            raise PeriodicOracleError(f"{case['id']} cell metadata mismatch")
        role = case.get("oracle_role")
        if role == "primary-neutral-full-model":
            primary_count += 1
            if int(case["molecular_charge"]) != 0:
                raise PeriodicOracleError(f"{case['id']} primary tblite cell must be neutral")
        elif role == "diagnostic-unbackgrounded-charged":
            diagnostic_count += 1
            if int(case["molecular_charge"]) == 0:
                raise PeriodicOracleError(f"{case['id']} charged diagnostic must be charged")
        else:
            raise PeriodicOracleError(f"{case['id']} has unknown oracle role")

        golden_path = repository_path(case["golden"])
        if sha256_file(golden_path) != case.get("golden_sha256"):
            raise PeriodicOracleError(f"{case['id']} golden SHA-256 mismatch")
        golden = load_json(golden_path)
        if (
            golden.get("schema") != GOLDEN_SCHEMA
            or golden.get("case_id") != case["id"]
            or golden.get("oracle_role") != role
        ):
            raise PeriodicOracleError(f"{case['id']} golden identity mismatch")
        properties = golden.get("properties")
        if not isinstance(properties, dict):
            raise PeriodicOracleError(f"{case['id']} properties must be an object")
        finite_number(properties.get("energy_hartree"), f"{case['id']} energy")
        gradient = finite_vector(
            properties.get("gradient_hartree_per_bohr"),
            3 * int(case["atom_count"]),
            f"{case['id']} gradient",
        )
        forces = finite_vector(
            properties.get("forces_hartree_per_bohr"),
            len(gradient),
            f"{case['id']} forces",
        )
        if forces != [-value for value in gradient]:
            raise PeriodicOracleError(f"{case['id']} force sign drifted")
        strain = finite_vector(
            properties.get("strain_derivatives_hartree"), 9, f"{case['id']} strain"
        )
        if golden.get("source_output_sha256") != sha256_json(properties):
            raise PeriodicOracleError(f"{case['id']} normalized output hash mismatch")
        provenance = golden.get("provenance")
        if not isinstance(provenance, dict):
            raise PeriodicOracleError(f"{case['id']} provenance must be an object")
        if (
            provenance.get("accuracy") != ACCURACY
            or provenance.get("command_template") != command_template()
            or provenance.get("engine") != "tblite"
            or provenance.get("executable_sha256") != reference["executable_sha256"]
            or provenance.get("input") != case["input"]
            or provenance.get("input_sha256") != case["input_sha256"]
            or provenance.get("source_revision") != reference["revision"]
            or provenance.get("runtime", {}).get("libtblite", {}).get("sha256")
            != runtime["libtblite_sha256"]
        ):
            raise PeriodicOracleError(f"{case['id']} oracle provenance mismatch")
        if case.get("strain_finite_difference", False):
            check_strain_evidence(golden, strain)
        elif "strain_finite_difference" in golden:
            raise PeriodicOracleError(f"{case['id']} has unrequested strain evidence")
    if primary_count < 4 or diagnostic_count < 1:
        raise PeriodicOracleError("periodic corpus lacks the reviewed case roles")
    check_analytic_background(manifest)
    print(
        f"periodic GFN2 corpus check passed: {primary_count} primary, "
        f"{diagnostic_count} diagnostic, "
        f"{len(manifest['analytic_background_cases'])} analytic background"
    )  # noqa: T201 - CLI result


def compare(
    manifest_path: Path, actual_dir: Path, names: list[str] | None
) -> None:
    """Compare independently generated normalized results with committed goldens."""
    manifest = load_json(manifest_path)
    compared = 0
    for case in selected_cases(manifest, names):
        if case["oracle_role"] == "analytic-background":
            continue
        expected = load_json(repository_path(case["golden"]))
        actual_path = actual_dir / f"{case['id']}.json"
        actual = load_json(actual_path)
        if actual.get("case_id") != case["id"]:
            raise PeriodicOracleError(f"{actual_path} has the wrong case ID")
        tolerances = case["tolerances"]
        for property_name in (
            "energy_hartree",
            "forces_hartree_per_bohr",
            "strain_derivatives_hartree",
        ):
            expected_value = expected["properties"][property_name]
            actual_value = actual["properties"][property_name]
            expected_values = expected_value if isinstance(expected_value, list) else [expected_value]
            actual_values = actual_value if isinstance(actual_value, list) else [actual_value]
            if len(expected_values) != len(actual_values):
                raise PeriodicOracleError(f"{case['id']} {property_name} extent mismatch")
            tolerance = finite_number(tolerances[property_name], f"{case['id']} tolerance")
            maximum = max(
                abs(finite_number(lhs, property_name) - finite_number(rhs, property_name))
                for lhs, rhs in zip(expected_values, actual_values, strict=True)
            )
            if maximum > tolerance:
                raise PeriodicOracleError(
                    f"{case['id']} {property_name} maximum error {maximum} > {tolerance}"
                )
        compared += 1
    print(f"periodic GFN2 comparison passed: {compared} cases")  # noqa: T201


def parser() -> argparse.ArgumentParser:
    """Build the standalone periodic oracle CLI."""
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subcommands = command.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check", help="verify committed evidence offline")
    generate_parser = subcommands.add_parser(
        "generate", help="generate tblite results into a separate directory"
    )
    generate_parser.add_argument("--executable", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--case", action="append", dest="cases")
    compare_parser = subcommands.add_parser(
        "compare", help="compare a generated directory with committed goldens"
    )
    compare_parser.add_argument("--actual-dir", type=Path, required=True)
    compare_parser.add_argument("--case", action="append", dest="cases")
    return command


def main() -> int:
    """Run the requested periodic conformance operation."""
    arguments = parser().parse_args()
    try:
        if arguments.command == "check":
            check(arguments.manifest)
        elif arguments.command == "generate":
            generate(
                arguments.manifest,
                arguments.executable,
                arguments.output_dir,
                arguments.cases,
            )
        elif arguments.command == "compare":
            compare(
                arguments.manifest, arguments.actual_dir, arguments.cases
            )
        else:  # pragma: no cover - argparse enforces the command set.
            raise PeriodicOracleError(f"unknown command: {arguments.command}")
    except PeriodicOracleError as exc:
        print(f"periodic GFN2 oracle error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
