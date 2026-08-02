#!/usr/bin/env python3
"""Create and compare reproducible GFN-xTB conformance data.

The committed corpus uses atomic units throughout.  tblite reports gradients,
while gpuxtb exposes forces, so this tool performs the single authoritative
conversion ``force = -gradient`` when normalizing reference output.
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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "conformance" / "manifest.json"


class ConformanceError(RuntimeError):
    """An actionable corpus, reference execution, or comparison failure."""


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and report its path on malformed input."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConformanceError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConformanceError(f"expected a JSON object in {path}")
    return value


def dump_json(path: Path, value: dict[str, Any]) -> None:
    """Write canonical, review-friendly JSON used by committed golden files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_manifest_path(manifest_path: Path, relative_path: str) -> Path:
    """Resolve corpus paths relative to the repository, not the current shell."""
    path = Path(relative_path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def selected_cases(
    manifest: dict[str, Any], names: list[str] | None
) -> list[dict[str, Any]]:
    """Select named cases while rejecting typos and duplicate manifest IDs."""
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ConformanceError("manifest 'cases' must be an array")
    by_name: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ConformanceError(
                "each manifest case must be an object with a string 'id'"
            )
        case_id = case["id"]
        if case_id in by_name:
            raise ConformanceError(f"duplicate case ID in manifest: {case_id}")
        by_name[case_id] = case
    if not names:
        return list(by_name.values())
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ConformanceError(f"unknown case ID(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def normalize_tblite_output(
    raw: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    """Convert tblite JSON energy/gradient/virial output into gpuxtb conventions."""
    try:
        energy = float(raw["energy"])
        gradient = [float(value) for value in raw["gradient"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConformanceError(
            "tblite output must contain numeric 'energy' and 'gradient'"
        ) from exc
    expected_components = 3 * int(case["atom_count"])
    if len(gradient) != expected_components:
        raise ConformanceError(
            f"case {case['id']} has {len(gradient)} gradient components; "
            f"expected {expected_components}"
        )
    properties: dict[str, Any] = {
        "energy_hartree": energy,
        "forces_hartree_per_bohr": [-value for value in gradient],
        "gradient_hartree_per_bohr": gradient,
    }
    if "virial" in raw:
        try:
            properties["virial_hartree"] = [float(value) for value in raw["virial"]]
        except (TypeError, ValueError) as exc:
            raise ConformanceError("tblite 'virial' must be a numeric array") from exc
    return properties


def _numeric_matrix(
    value: Any, rows: int, columns: int, property_name: str
) -> list[list[float]]:
    """Validate and normalize one atom-major matrix from reference JSON."""
    if not isinstance(value, list) or len(value) != rows:
        raise ConformanceError(f"xtb '{property_name}' must contain {rows} atom rows")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ConformanceError(
                f"xtb '{property_name}' rows must contain {columns} components"
            )
        try:
            matrix.append([float(component) for component in row])
        except (TypeError, ValueError) as exc:
            raise ConformanceError(
                f"xtb '{property_name}' must contain only numeric components"
            ) from exc
    return matrix


def parse_xtb_gradient(text: str, atom_count: int) -> tuple[float, list[float]]:
    """Extract high-precision energy and Cartesian gradient from xtb's file.

    xtb's JSON rounds energies to eight decimal places, while its ``gradient``
    artifact retains enough digits for the corpus tolerance.  The file places
    one coordinate row per atom before the corresponding gradient rows.
    """
    lines = text.splitlines()
    cycle_index = next(
        (index for index, line in enumerate(lines) if "SCF energy" in line), None
    )
    if cycle_index is None:
        raise ConformanceError("xtb gradient file has no SCF energy header")
    match = re.search(
        r"SCF\s+energy\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
        lines[cycle_index],
    )
    if match is None:
        raise ConformanceError("cannot parse energy from xtb gradient header")
    energy = float(match.group(1).replace("D", "E").replace("d", "e"))
    gradient_lines = lines[
        cycle_index + 1 + atom_count : cycle_index + 1 + 2 * atom_count
    ]
    if len(gradient_lines) != atom_count:
        raise ConformanceError(
            f"xtb gradient file contains {len(gradient_lines)} gradient rows; "
            f"expected {atom_count}"
        )
    gradient: list[float] = []
    for line in gradient_lines:
        fields = line.split()
        if len(fields) < 3:
            raise ConformanceError(f"malformed xtb gradient row: {line!r}")
        try:
            gradient.extend(
                float(field.replace("D", "E").replace("d", "e")) for field in fields[:3]
            )
        except ValueError as exc:
            raise ConformanceError(f"malformed xtb gradient row: {line!r}") from exc
    return energy, gradient


def normalize_xtb_output(
    raw: dict[str, Any], gradient_text: str, case: dict[str, Any]
) -> dict[str, Any]:
    """Normalize xtb energy, gradient, and atom-resolved GFN2 SCC moments."""
    atom_count = int(case["atom_count"])
    energy, gradient = parse_xtb_gradient(gradient_text, atom_count)
    try:
        charges = [float(value) for value in raw["partial charges"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConformanceError("xtb JSON has no numeric 'partial charges'") from exc
    if len(charges) != atom_count:
        raise ConformanceError(
            f"xtb JSON contains {len(charges)} charges; expected {atom_count}"
        )
    dipoles = _numeric_matrix(
        raw.get("atomic dipole moments"), atom_count, 3, "atomic dipole moments"
    )
    quadrupoles = _numeric_matrix(
        raw.get("atomic quadrupole moments"),
        atom_count,
        6,
        "atomic quadrupole moments",
    )
    try:
        reported_unpaired = int(raw["number of unpaired electrons"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConformanceError(
            "xtb JSON has no integer 'number of unpaired electrons'"
        ) from exc
    if reported_unpaired != int(case["unpaired_electrons"]):
        raise ConformanceError(
            f"xtb reported {reported_unpaired} unpaired electrons for {case['id']}; "
            f"expected {case['unpaired_electrons']}"
        )
    return {
        "atomic_dipoles_e_bohr": dipoles,
        # xtb emits the symmetric tensor in xx, xy, yy, xz, yz, zz order.
        "atomic_quadrupoles_e_bohr2": quadrupoles,
        "energy_hartree": energy,
        "forces_hartree_per_bohr": [-value for value in gradient],
        "gradient_hartree_per_bohr": gradient,
        "partial_charges_e": charges,
    }


def sha256_json(value: Any) -> str:
    """Hash canonical JSON independent of temporary paths and whitespace."""
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_golden(
    manifest: dict[str, Any],
    case: dict[str, Any],
    raw: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the versioned golden schema shared by imported and live references."""
    return {
        "case_id": case["id"],
        "method": manifest["method"],
        "molecular_charge": case["molecular_charge"],
        "provenance": provenance,
        "properties": normalize_tblite_output(raw, case),
        "schema_version": manifest["golden_schema_version"],
        "units": manifest["units"],
        "unpaired_electrons": case["unpaired_electrons"],
    }


def canonical_xtb_golden(
    manifest: dict[str, Any],
    case: dict[str, Any],
    properties: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build a canonical golden containing xtb's atom-resolved SCC state."""
    return {
        "case_id": case["id"],
        "method": manifest["method"],
        "molecular_charge": case["molecular_charge"],
        "provenance": provenance,
        "properties": properties,
        "schema_version": manifest["golden_schema_version"],
        "units": manifest["units"],
        "unpaired_electrons": case["unpaired_electrons"],
    }


def check_manifest(manifest_path: Path) -> None:
    """Validate schema invariants, source hashes, and committed corpus hashes."""
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("golden_schema_version") != 1
    ):
        raise ConformanceError(
            "only manifest and golden schema version 1 are supported"
        )
    if manifest.get("method") != "GFN2-xTB":
        raise ConformanceError("the initial corpus must use method 'GFN2-xTB'")
    units = manifest.get("units", {})
    expected_units = {
        "coordinates": "bohr",
        "energy": "hartree",
        "forces": "hartree/bohr",
        "gradient": "hartree/bohr",
    }
    if any(units.get(key) != value for key, value in expected_units.items()):
        raise ConformanceError(f"manifest units must include {expected_units}")

    cases = selected_cases(manifest, None)
    for case in cases:
        input_path = resolve_manifest_path(manifest_path, case["input"])
        golden_path = resolve_manifest_path(manifest_path, case["golden"])
        for label, path, digest_key in (
            ("input", input_path, "input_sha256"),
            ("golden", golden_path, "golden_sha256"),
        ):
            if not path.is_file():
                raise ConformanceError(f"case {case['id']} {label} is missing: {path}")
            actual_digest = sha256_file(path)
            if actual_digest != case.get(digest_key):
                raise ConformanceError(
                    f"case {case['id']} {label} SHA-256 mismatch: "
                    f"manifest={case.get(digest_key)} actual={actual_digest}"
                )
        golden = load_json(golden_path)
        if golden.get("case_id") != case["id"]:
            raise ConformanceError(f"golden {golden_path} has the wrong case_id")
        if golden.get("method") != manifest["method"] or golden.get("units") != units:
            raise ConformanceError(
                f"golden {golden_path} has inconsistent method or units"
            )
        if golden.get("molecular_charge") != case["molecular_charge"]:
            raise ConformanceError(
                f"golden {golden_path} has the wrong molecular charge"
            )
        if golden.get("unpaired_electrons") != case["unpaired_electrons"]:
            raise ConformanceError(f"golden {golden_path} has the wrong spin state")
        provenance = golden.get("provenance", {})
        reference_engine = case.get("reference_engine", "tblite")
        if provenance.get("engine") != reference_engine:
            raise ConformanceError(
                f"golden {golden_path} has the wrong reference engine"
            )
        if (
            provenance.get("source_revision")
            != manifest["reference_engines"][reference_engine]["revision"]
        ):
            raise ConformanceError(
                f"golden {golden_path} has the wrong {reference_engine} revision"
            )
        expected_output_hash = case.get(
            "reference_output_sha256", case.get("upstream_output_sha256")
        )
        if provenance.get("source_output_sha256") != expected_output_hash:
            raise ConformanceError(
                f"golden {golden_path} has the wrong reference output hash"
            )
        properties = golden.get("properties", {})
        forces = properties.get("forces_hartree_per_bohr")
        gradient = properties.get("gradient_hartree_per_bohr")
        expected_components = 3 * int(case["atom_count"])
        if not isinstance(forces, list) or len(forces) != expected_components:
            raise ConformanceError(f"golden {golden_path} has the wrong force shape")
        if not isinstance(gradient, list) or len(gradient) != expected_components:
            raise ConformanceError(f"golden {golden_path} has the wrong gradient shape")
        numeric_values = [properties.get("energy_hartree"), *forces, *gradient]
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ConformanceError(
                f"golden {golden_path} contains a non-finite required value"
            )
        if any(
            not math.isclose(float(force), -float(grad), abs_tol=0.0)
            for force, grad in zip(forces, gradient)
        ):
            raise ConformanceError(f"golden {golden_path} violates force = -gradient")
        if reference_engine == "xtb":
            if provenance.get("source_output_sha256") != sha256_json(properties):
                raise ConformanceError(
                    f"golden {golden_path} does not match its normalized xtb output hash"
                )
            expected_shapes = {
                "partial_charges_e": (int(case["atom_count"]),),
                "atomic_dipoles_e_bohr": (int(case["atom_count"]), 3),
                "atomic_quadrupoles_e_bohr2": (int(case["atom_count"]), 6),
            }
            for property_name, shape in expected_shapes.items():
                _validate_nested_shape(
                    properties.get(property_name), shape, property_name, golden_path
                )
    print(f"conformance manifest OK: {len(cases)} cases")


def _validate_nested_shape(
    value: Any, shape: tuple[int, ...], property_name: str, path: Path
) -> None:
    """Check a numeric nested-list shape and reject non-finite SCC state."""
    if not shape:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ConformanceError(
                f"golden {path} property {property_name} is not numeric"
            ) from exc
        if not math.isfinite(numeric):
            raise ConformanceError(
                f"golden {path} property {property_name} is non-finite"
            )
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ConformanceError(
            f"golden {path} property {property_name} has the wrong shape"
        )
    for component in value:
        _validate_nested_shape(component, shape[1:], property_name, path)


def executable_version(executable: Path) -> str:
    """Capture reference version text without assuming a specific release format."""
    completed = subprocess.run(
        [str(executable), "--version"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    text = completed.stdout.strip()
    return text if text else f"--version exited with status {completed.returncode}"


def tblite_command(
    executable: Path, case: dict[str, Any], input_path: Path, output_path: Path
) -> list[str]:
    """Construct the command used by tblite's own validation benchmarks."""
    command = [
        str(executable),
        str(input_path),
        "--no-restart",
        "--method",
        "gfn2",
        "--grad",
        str(output_path.with_suffix(".txt")),
    ]
    charge = int(case["molecular_charge"])
    if charge:
        command.extend(["--charge", f"{charge:+d}"])
    unpaired_electrons = int(case["unpaired_electrons"])
    if unpaired_electrons:
        command.extend(["--spin", str(unpaired_electrons)])
    command.extend(["--json", str(output_path)])
    return command


def generate_with_tblite(
    manifest_path: Path,
    executable: Path,
    output_dir: Path,
    case_names: list[str] | None,
) -> None:
    """Run a tblite executable and emit normalized, provenance-rich golden JSON."""
    manifest = load_json(manifest_path)
    cases = selected_cases(manifest, case_names)
    resolved_executable = Path(shutil.which(str(executable)) or executable).resolve()
    if not resolved_executable.is_file():
        raise ConformanceError(f"tblite executable does not exist: {executable}")
    version = executable_version(resolved_executable)
    executable_digest = sha256_file(resolved_executable)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gpuxtb-conformance-") as temporary:
        work = Path(temporary)
        for case in cases:
            input_path = resolve_manifest_path(manifest_path, case["input"])
            raw_output = work / f"{case['id']}.json"
            command = tblite_command(resolved_executable, case, input_path, raw_output)
            completed = subprocess.run(
                command,
                cwd=work,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0:
                raise ConformanceError(
                    f"tblite failed for {case['id']} with status {completed.returncode}:\n"
                    f"{completed.stdout}"
                )
            raw = load_json(raw_output)
            provenance = {
                # Store the reviewed template rather than temporary absolute
                # paths, which would make otherwise identical runs differ.
                "command": manifest["reference_engines"]["tblite"][
                    "cli_command_template"
                ],
                "engine": "tblite",
                "executable_sha256": executable_digest,
                "executable_version": version,
                "generation_mode": "live-cli",
                "input": case["input"],
            }
            destination = output_dir / f"{case['id']}.json"
            dump_json(destination, canonical_golden(manifest, case, raw, provenance))
            print(f"generated {destination}")


def xtb_command(executable: Path, case: dict[str, Any]) -> list[str]:
    """Construct a deterministic single-threaded GFN2 gradient calculation."""
    return [
        str(executable),
        "coord",
        "--gfn",
        "2",
        "--grad",
        "--json",
        "--norestart",
        "--chrg",
        str(int(case["molecular_charge"])),
        "--uhf",
        str(int(case["unpaired_electrons"])),
        "-P",
        "1",
    ]


def generate_with_xtb(
    manifest_path: Path,
    executable: Path,
    output_dir: Path,
    case_names: list[str] | None,
) -> None:
    """Run xtb and emit normalized energy, force, and SCC-state goldens."""
    manifest = load_json(manifest_path)
    cases = selected_cases(manifest, case_names)
    resolved_executable = Path(shutil.which(str(executable)) or executable).resolve()
    if not resolved_executable.is_file():
        raise ConformanceError(f"xtb executable does not exist: {executable}")
    version = executable_version(resolved_executable)
    executable_digest = sha256_file(resolved_executable)
    xtb_reference = manifest["reference_engines"]["xtb"]
    expected_version = str(xtb_reference["version"])
    expected_revision_prefix = str(xtb_reference["revision"])[:7]
    if (
        f"xtb version {expected_version}" not in version
        or f"({expected_revision_prefix})" not in version
    ):
        raise ConformanceError(
            "xtb executable does not match the pinned oracle: expected "
            f"version {expected_version} ({expected_revision_prefix}), got:\n{version}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explicit one-thread settings avoid BLAS/OpenMP scheduling differences and
    # also prevent oversubscription when this command is later used from CI.
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "OMP_NUM_THREADS": "1",
            "OMP_STACKSIZE": "4G",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="gpuxtb-conformance-xtb-") as temporary:
        root = Path(temporary)
        for case in cases:
            work = root / case["id"]
            work.mkdir()
            input_path = resolve_manifest_path(manifest_path, case["input"])
            shutil.copyfile(input_path, work / "coord")
            command = xtb_command(resolved_executable, case)
            completed = subprocess.run(
                command,
                cwd=work,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0:
                raise ConformanceError(
                    f"xtb failed for {case['id']} with status {completed.returncode}:\n"
                    f"{completed.stdout}"
                )
            raw = load_json(work / "xtbout.json")
            try:
                gradient_text = (work / "gradient").read_text(encoding="utf-8")
            except OSError as exc:
                raise ConformanceError(
                    f"cannot read xtb gradient for {case['id']}: {exc}"
                ) from exc
            properties = normalize_xtb_output(raw, gradient_text, case)
            provenance = {
                "command": xtb_reference["cli_command_template"],
                "engine": "xtb",
                "executable_sha256": executable_digest,
                "executable_version": version,
                "generation_mode": "live-cli",
                "input": case["input"],
                # Hash parsed scientific output, excluding xtb's absolute
                # executable path and other non-reproducible JSON metadata.
                "source_output_sha256": sha256_json(properties),
                "source_revision": xtb_reference["revision"],
            }
            destination = output_dir / f"{case['id']}.json"
            dump_json(
                destination,
                canonical_xtb_golden(manifest, case, properties, provenance),
            )
            print(f"generated {destination}")


def git_revision(source_root: Path) -> str:
    """Read the exact source revision used for snapshot import."""
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ConformanceError(
            f"cannot read git revision for {source_root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def import_tblite_snapshot(
    manifest_path: Path, source_root: Path, allow_revision_mismatch: bool
) -> None:
    """Recreate committed inputs/goldens from tblite's pinned validation assets."""
    manifest = load_json(manifest_path)
    source = manifest["reference_engines"]["tblite"]
    actual_revision = git_revision(source_root)
    expected_revision = source["revision"]
    if actual_revision != expected_revision and not allow_revision_mismatch:
        raise ConformanceError(
            f"tblite revision mismatch: expected {expected_revision}, got {actual_revision}; "
            "use --allow-revision-mismatch only for an intentional corpus update"
        )

    cases = [
        case
        for case in selected_cases(manifest, None)
        if case.get("reference_engine", "tblite") == "tblite"
    ]
    for case in cases:
        upstream_directory = source_root / case["upstream_validation_case"]
        upstream_input = upstream_directory / "coord"
        upstream_output = upstream_directory / "gfn2-xtb.json"
        for label, path, expected_digest in (
            ("input", upstream_input, case["upstream_input_sha256"]),
            ("output", upstream_output, case["upstream_output_sha256"]),
        ):
            actual_digest = sha256_file(path)
            if actual_digest != expected_digest and not allow_revision_mismatch:
                raise ConformanceError(
                    f"tblite snapshot {label} hash mismatch for {case['id']}: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
        input_path = resolve_manifest_path(manifest_path, case["input"])
        golden_path = resolve_manifest_path(manifest_path, case["golden"])
        input_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream_input, input_path)
        raw = load_json(upstream_output)
        provenance = {
            "command": source["snapshot_command_template"],
            "engine": "tblite",
            "generation_mode": "upstream-validation-snapshot",
            "source_output_sha256": sha256_file(upstream_output),
            "source_path": case["upstream_validation_case"] + "/gfn2-xtb.json",
            "source_revision": actual_revision,
        }
        dump_json(golden_path, canonical_golden(manifest, case, raw, provenance))
        print(f"imported {case['id']}")
    print("snapshot imported; update manifest hashes before running 'check'")


def actual_properties(raw: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Accept canonical golden, raw tblite, or a minimal gpuxtb result object."""
    if isinstance(raw.get("properties"), dict):
        return raw["properties"]
    if "energy" in raw and "gradient" in raw:
        return normalize_tblite_output(raw, case)
    energy = raw.get("energy_hartree", raw.get("energy"))
    forces = raw.get("forces_hartree_per_bohr", raw.get("forces"))
    if energy is None or forces is None:
        raise ConformanceError(
            "actual JSON must be canonical golden, raw tblite output, or contain "
            "energy_hartree and forces_hartree_per_bohr"
        )
    properties = {
        "energy_hartree": float(energy),
        "forces_hartree_per_bohr": [float(value) for value in forces],
    }
    for property_name in (
        "partial_charges_e",
        "atomic_dipoles_e_bohr",
        "atomic_quadrupoles_e_bohr2",
    ):
        if property_name in raw:
            properties[property_name] = raw[property_name]
    return properties


def compare_values(
    case_id: str,
    property_name: str,
    expected: Any,
    actual: Any,
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    """Compare a scalar or nested numeric array and report the worst component."""
    expected_values = _flatten_numeric(expected, property_name)
    actual_values = _flatten_numeric(actual, property_name)
    if len(expected_values) != len(actual_values):
        return (
            False,
            f"{case_id} {property_name}: shape {len(actual_values)} != {len(expected_values)}",
        )
    worst_index = 0
    worst_error = -1.0
    worst_limit = 0.0
    passed = True
    for index, (expected_value, actual_value) in enumerate(
        zip(expected_values, actual_values)
    ):
        expected_float = float(expected_value)
        actual_float = float(actual_value)
        error = abs(actual_float - expected_float)
        limit = atol + rtol * abs(expected_float)
        if not math.isfinite(actual_float) or error > limit:
            passed = False
        if error > worst_error:
            worst_index, worst_error, worst_limit = index, error, limit
    location = (
        "scalar" if not isinstance(expected, list) else f"component {worst_index}"
    )
    return passed, (
        f"{case_id} {property_name}: max_abs_error={worst_error:.6e} "
        f"limit={worst_limit:.6e} at {location}"
    )


def _flatten_numeric(value: Any, property_name: str) -> list[float]:
    """Flatten a scalar or nested list while preserving deterministic order."""
    if isinstance(value, list):
        flattened: list[float] = []
        for component in value:
            flattened.extend(_flatten_numeric(component, property_name))
        return flattened
    try:
        return [float(value)]
    except (TypeError, ValueError) as exc:
        raise ConformanceError(f"{property_name} must contain numeric values") from exc


def compare_directory(
    manifest_path: Path, actual_dir: Path, case_names: list[str] | None
) -> None:
    """Compare per-case JSON results against committed energy and force goldens."""
    manifest = load_json(manifest_path)
    failures: list[str] = []
    for case in selected_cases(manifest, case_names):
        golden_path = resolve_manifest_path(manifest_path, case["golden"])
        golden = load_json(golden_path)
        expected = golden["properties"]
        actual_path = actual_dir / f"{case['id']}.json"
        actual_document = load_json(actual_path)
        actual = actual_properties(actual_document, case)
        expected_engine = golden.get("provenance", {}).get("engine")
        actual_engine = actual_document.get("provenance", {}).get("engine")
        tolerances = manifest["tolerances"]
        if (
            actual_engine in manifest["reference_engines"]
            and expected_engine in manifest["reference_engines"]
            and actual_engine != expected_engine
        ):
            # xtb and tblite are independent implementations with small force
            # differences. This looser gate applies only when both documents
            # explicitly identify distinct reference engines; gpuxtb results
            # continue to use the primary, stricter acceptance tolerances.
            tolerances = manifest["cross_engine_tolerances"]
        compared_properties = [
            ("energy_hartree", "energy"),
            ("forces_hartree_per_bohr", "forces"),
        ]
        compared_properties.extend(
            (property_name, tolerance_name)
            for property_name, tolerance_name in (
                ("partial_charges_e", "charges"),
                ("atomic_dipoles_e_bohr", "atomic_dipoles"),
                ("atomic_quadrupoles_e_bohr2", "atomic_quadrupoles"),
            )
            if property_name in expected
        )
        for property_name, tolerance_name in compared_properties:
            if property_name not in actual:
                failures.append(f"{case['id']} is missing {property_name}")
                print(f"FAIL {failures[-1]}")
                continue
            tolerance = tolerances[tolerance_name]
            passed, message = compare_values(
                case["id"],
                property_name,
                expected[property_name],
                actual[property_name],
                float(tolerance["atol"]),
                float(tolerance["rtol"]),
            )
            print(("PASS " if passed else "FAIL ") + message)
            if not passed:
                failures.append(message)
    if failures:
        raise ConformanceError(f"{len(failures)} conformance comparison(s) failed")


def build_parser() -> argparse.ArgumentParser:
    """Define the stable command-line entry used locally and by future CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "check", help="verify manifest, corpus hashes, units, and shapes"
    )

    generate = subparsers.add_parser(
        "generate", help="generate normalized goldens with tblite"
    )
    generate.add_argument("--executable", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--case", action="append", dest="cases")

    generate_xtb = subparsers.add_parser(
        "generate-xtb",
        help="generate normalized energy, force, and SCC-state goldens with xtb",
    )
    generate_xtb.add_argument("--executable", type=Path, required=True)
    generate_xtb.add_argument("--output-dir", type=Path, required=True)
    generate_xtb.add_argument("--case", action="append", dest="cases")

    snapshot = subparsers.add_parser(
        "import-tblite-snapshot",
        help="recreate the corpus from a pinned tblite source tree",
    )
    snapshot.add_argument("--source-root", type=Path, required=True)
    snapshot.add_argument("--allow-revision-mismatch", action="store_true")

    compare = subparsers.add_parser(
        "compare", help="compare a directory of per-case JSON results"
    )
    compare.add_argument("--actual-dir", type=Path, required=True)
    compare.add_argument("--case", action="append", dest="cases")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the CLI, converting expected user errors into a concise nonzero exit."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            check_manifest(args.manifest)
        elif args.command == "generate":
            generate_with_tblite(
                args.manifest, args.executable, args.output_dir, args.cases
            )
        elif args.command == "generate-xtb":
            generate_with_xtb(
                args.manifest, args.executable, args.output_dir, args.cases
            )
        elif args.command == "import-tblite-snapshot":
            import_tblite_snapshot(
                args.manifest, args.source_root, args.allow_revision_mismatch
            )
        elif args.command == "compare":
            compare_directory(args.manifest, args.actual_dir, args.cases)
        else:  # pragma: no cover - argparse guarantees the command choices.
            raise AssertionError(args.command)
    except (ConformanceError, KeyError, OSError) as exc:
        print(f"conformance error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
