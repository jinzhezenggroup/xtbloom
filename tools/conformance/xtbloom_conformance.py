#!/usr/bin/env python3
"""Create and compare reproducible GFN-xTB conformance data.

The committed corpus uses atomic units throughout.  tblite reports gradients,
while xtbloom exposes forces, so this tool performs the single authoritative
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "conformance" / "manifest.json"
PRIMARY_ORACLE_ACCURACY = 1.0e-4
PRIMARY_ORACLE_ACCURACY_TEXT = "0.0001"
# One atomic unit of electric field (Hartree per elementary charge per bohr)
# expressed in V/angstrom; the tblite --efield flag takes V/angstrom.
ELECTRIC_FIELD_VAA_PER_AU = 0.019446903964791384
# Optional tblite command token describing a uniform electric field attachment.
EFIELD_COMMAND_TOKEN = "[--efield {efield_vperangstrom}]"


class ConformanceError(RuntimeError):
    """An actionable corpus, reference execution, or comparison failure."""


def reference_accuracy(reference: dict[str, Any], engine: str) -> str:
    """Validate and return the reviewed CLI spelling of the primary accuracy.

    Both reference engines use the same numerical setting.  Keeping the text
    token centralized prevents a float formatter or a stale command template
    from silently regenerating a looser SCC oracle.
    """
    value = reference.get("accuracy")
    if type(value) not in (int, float) or float(value) != PRIMARY_ORACLE_ACCURACY:
        raise ConformanceError(
            f"manifest {engine} accuracy must be {PRIMARY_ORACLE_ACCURACY_TEXT}"
        )
    return PRIMARY_ORACLE_ACCURACY_TEXT


def accepted_tblite_command_templates(
    reference: dict[str, Any], case: dict[str, Any]
) -> list[list[str]]:
    """Return provenance templates valid for one tblite-backed case.

    Field-free goldens may predate the optional ``--efield`` token in the
    manifest template. A field case must retain that token because it is the
    defining interaction of the oracle calculation.
    """
    expected = reference.get("cli_command_template")
    if not isinstance(expected, list) or not all(
        isinstance(token, str) for token in expected
    ):
        raise ConformanceError(
            "manifest tblite command template must be a string array"
        )
    if case.get("efield") is not None and EFIELD_COMMAND_TOKEN not in expected:
        raise ConformanceError(
            f"case {case['id']} has an electric field but the tblite template "
            "does not contain --efield"
        )
    accepted = [expected]
    if case.get("efield") is None and EFIELD_COMMAND_TOKEN in expected:
        accepted.append([token for token in expected if token != EFIELD_COMMAND_TOKEN])
    return accepted


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
    """Convert tblite JSON energy/gradient/virial output into xtbloom conventions."""
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
    value: object, rows: int, columns: int, property_name: str
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


QMMM_INPUT_UNITS = {
    "point_charge_gammas": "hartree",
    "point_charge_positions": "bohr",
    "point_charges": "elementary_charge",
    "qm_positions": "bohr",
}
QMMM_MATERIALIZATION_SCHEMA = "xtbloom-xtb-pcem-cli-v1"
ELEMENT_SYMBOLS = (
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
)


def load_turbomole_coord(path: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Load the small atomic-unit ``$coord`` subset used by the corpus.

    The conformance inputs may contain unrelated Turbomole directives after
    the coordinate block (for example ``$eht charge=+1``).  The public C API
    runner obtains charge and spin from the manifest, so this parser reads
    exactly the atom rows between ``$coord`` and the next directive and rejects
    malformed or non-finite coordinates instead of silently guessing.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConformanceError(
            f"cannot read Turbomole coordinate file {path}: {exc}"
        ) from exc

    symbol_numbers = {
        symbol.lower(): number
        for number, symbol in enumerate(ELEMENT_SYMBOLS)
        if symbol
    }
    in_coordinates = False
    found_coordinates = False
    atomic_numbers: list[int] = []
    positions: list[list[float]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_coordinates:
            if stripped.lower() == "$coord":
                in_coordinates = True
                found_coordinates = True
            continue
        if stripped.startswith("$"):
            break
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 4:
            raise ConformanceError(
                f"Turbomole input {path}:{line_number} must contain x y z symbol"
            )
        try:
            position = [float(value) for value in fields[:3]]
        except ValueError as exc:
            raise ConformanceError(
                f"Turbomole input {path}:{line_number} has non-numeric coordinates"
            ) from exc
        if any(not math.isfinite(value) for value in position):
            raise ConformanceError(
                f"Turbomole input {path}:{line_number} has non-finite coordinates"
            )
        symbol = fields[3].lower()
        if symbol not in symbol_numbers:
            raise ConformanceError(
                f"Turbomole input {path}:{line_number} has unsupported "
                f"element {fields[3]!r}"
            )
        positions.append(position)
        atomic_numbers.append(symbol_numbers[symbol])

    if not found_coordinates:
        raise ConformanceError(f"Turbomole input {path} has no $coord block")
    expected_atoms = int(case["atom_count"])
    if len(atomic_numbers) != expected_atoms:
        raise ConformanceError(
            f"case {case['id']} has {len(atomic_numbers)} coordinate rows; "
            f"expected {expected_atoms}"
        )
    return {"atomic_numbers": atomic_numbers, "positions_bohr": positions}


def _atomic_numbers(
    value: object, count: int, property_name: str, path: Path
) -> list[int]:
    """Validate integral atomic numbers supported by the pinned GFN2 model."""
    if not isinstance(value, list) or len(value) != count:
        raise ConformanceError(f"QMMM input {path} has invalid {property_name} shape")
    numbers: list[int] = []
    for number in value:
        if type(number) is not int or not 1 <= number < len(ELEMENT_SYMBOLS):
            raise ConformanceError(
                f"QMMM input {path} has unsupported {property_name} value {number!r}"
            )
        numbers.append(number)
    return numbers


def load_qmmm_input(
    path: Path,
    case: dict[str, Any],
    hardness_by_atomic_number: dict[str, Any],
) -> dict[str, Any]:
    """Load and validate the versioned QM plus external-point-charge input."""
    document = load_json(path)
    if document.get("schema_version") != 1:
        raise ConformanceError(f"QMMM input {path} must use schema version 1")
    if document.get("case_id") != case["id"]:
        raise ConformanceError(f"QMMM input {path} has the wrong case_id")
    if document.get("method") != "GFN2-xTB":
        raise ConformanceError(f"QMMM input {path} must use GFN2-xTB")
    if document.get("units") != QMMM_INPUT_UNITS:
        raise ConformanceError(f"QMMM input {path} has inconsistent units")

    qm = document.get("qm")
    point_charges = document.get("external_point_charges")
    if not isinstance(qm, dict) or not isinstance(point_charges, dict):
        raise ConformanceError(f"QMMM input {path} lacks QM or point-charge data")
    atom_count = int(case["atom_count"])
    point_charge_count = int(case["point_charge_count"])
    _validate_nested_shape(
        qm.get("positions_bohr"), (atom_count, 3), "qm.positions_bohr", path
    )
    atomic_numbers = _atomic_numbers(
        qm.get("atomic_numbers"), atom_count, "qm.atomic_numbers", path
    )
    symbols = qm.get("symbols")
    if (
        not isinstance(symbols, list)
        or len(symbols) != atom_count
        or any(type(symbol) is not str for symbol in symbols)
    ):
        raise ConformanceError(f"QMMM input {path} has invalid QM symbols")
    expected_symbols = [ELEMENT_SYMBOLS[number] for number in atomic_numbers]
    if symbols != expected_symbols:
        raise ConformanceError(
            f"QMMM input {path} QM symbols do not match atomic_numbers: "
            f"expected {expected_symbols}, got {symbols}"
        )
    if qm.get("molecular_charge") != case["molecular_charge"]:
        raise ConformanceError(f"QMMM input {path} has the wrong molecular charge")
    if qm.get("unpaired_electrons") != case["unpaired_electrons"]:
        raise ConformanceError(f"QMMM input {path} has the wrong spin state")

    for property_name in ("charges_e", "gammas_hartree"):
        _validate_nested_shape(
            point_charges.get(property_name),
            (point_charge_count,),
            f"external_point_charges.{property_name}",
            path,
        )
    _validate_nested_shape(
        point_charges.get("positions_bohr"),
        (point_charge_count, 3),
        "external_point_charges.positions_bohr",
        path,
    )
    gammas = [float(gamma) for gamma in point_charges["gammas_hartree"]]
    if any(gamma <= 0.0 for gamma in gammas):
        raise ConformanceError(f"QMMM input {path} contains a non-positive gamma")
    gamma_mode = point_charges.get("gamma_mode")
    source_numbers_value = point_charges.get("source_atomic_numbers")
    if gamma_mode == "explicit":
        if source_numbers_value is not None:
            raise ConformanceError(
                f"QMMM input {path} explicit gamma mode must not provide "
                "source atomic numbers"
            )
    elif gamma_mode == "element_hardness":
        source_numbers = _atomic_numbers(
            source_numbers_value,
            point_charge_count,
            "external_point_charges.source_atomic_numbers",
            path,
        )
        expected_gammas: list[float] = []
        for number in source_numbers:
            try:
                expected_gammas.append(float(hardness_by_atomic_number[str(number)]))
            except (KeyError, TypeError, ValueError) as exc:  # noqa: PERF203 - retain Z context
                raise ConformanceError(
                    f"QMMM input {path} has no pinned GFN2 hardness for Z={number}"
                ) from exc
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(gammas, expected_gammas, strict=True)
        ):
            raise ConformanceError(
                f"QMMM input {path} element-hardness gammas do not match "
                f"the pinned GFN2 values {expected_gammas}"
            )
    else:
        raise ConformanceError(
            f"QMMM input {path} gamma_mode must be 'explicit' or 'element_hardness'"
        )
    return document


def materialize_xtb_qmmm(document: dict[str, Any]) -> dict[str, str]:
    """Serialize a validated QMMM document into deterministic xTB CLI files."""
    qm = document["qm"]
    coord_lines = ["$coord"]
    for position, symbol in zip(qm["positions_bohr"], qm["symbols"], strict=True):
        coord_lines.append(
            " ".join([*(f"{float(value):.17g}" for value in position), symbol.lower()])
        )
    coord_lines.append("$end")

    point_charges = document["external_point_charges"]
    pcharge_lines = [str(len(point_charges["charges_e"]))]
    for charge, position, gamma in zip(
        point_charges["charges_e"],
        point_charges["positions_bohr"],
        point_charges["gammas_hartree"],
        strict=True,
    ):
        pcharge_lines.append(
            " ".join(
                [
                    f"{float(charge):.17g}",
                    *(f"{float(value):.17g}" for value in position),
                    f"{float(gamma):.17g}",
                ]
            )
        )
    return {
        "coord": "\n".join(coord_lines) + "\n",
        "pcharge": "\n".join(pcharge_lines) + "\n",
        "xcontrol": "$embedding\n input=pcharge\n gradient=pcgrad\n$end\n",
    }


def qmmm_materialization_provenance(document: dict[str, Any]) -> dict[str, Any]:
    """Describe the exact deterministic files passed to the xTB CLI."""
    return {
        "files_sha256": {
            filename: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for filename, content in materialize_xtb_qmmm(document).items()
        },
        "schema": QMMM_MATERIALIZATION_SCHEMA,
    }


def write_xtb_qmmm_files(work: Path, document: dict[str, Any]) -> dict[str, str]:
    """Write deterministic xTB inputs and return their content SHA-256 values."""
    record = qmmm_materialization_provenance(document)
    for filename, content in materialize_xtb_qmmm(document).items():
        (work / filename).write_text(content, encoding="utf-8")
    return record["files_sha256"]


def parse_xtb_pcgradient(text: str, point_charge_count: int) -> list[float]:
    """Parse xTB's three-column external point-charge gradient artifact."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if len(rows) != point_charge_count or any(len(row) != 3 for row in rows):
        raise ConformanceError(
            "xtb point-charge gradient has the wrong number of rows or columns"
        )
    try:
        return [
            float(component.replace("D", "E").replace("d", "e"))
            for row in rows
            for component in row
        ]
    except ValueError as exc:
        raise ConformanceError("xtb point-charge gradient is not numeric") from exc


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
    raw: dict[str, Any],
    gradient_text: str,
    case: dict[str, Any],
    pcgradient_text: str | None = None,
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
    properties = {
        "atomic_dipoles_e_bohr": dipoles,
        # xtb emits the symmetric tensor in xx, xy, yy, xz, yz, zz order.
        "atomic_quadrupoles_e_bohr2": quadrupoles,
        "energy_hartree": energy,
        "forces_hartree_per_bohr": [-value for value in gradient],
        "gradient_hartree_per_bohr": gradient,
        "partial_charges_e": charges,
    }
    point_charge_count = int(case.get("point_charge_count", 0))
    if point_charge_count:
        if pcgradient_text is None:
            raise ConformanceError("xtb produced no point-charge gradient artifact")
        pcgradient = parse_xtb_pcgradient(pcgradient_text, point_charge_count)
        properties["point_charge_forces_hartree_per_bohr"] = [
            -value for value in pcgradient
        ]
        properties["point_charge_gradient_hartree_per_bohr"] = pcgradient
    return properties


def sha256_json(value: object) -> str:
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
    qmmm_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical golden containing xtb's atom-resolved SCC state."""
    golden = {
        "case_id": case["id"],
        "method": manifest["method"],
        "molecular_charge": case["molecular_charge"],
        "provenance": provenance,
        "properties": properties,
        "schema_version": manifest["golden_schema_version"],
        "units": manifest["units"],
        "unpaired_electrons": case["unpaired_electrons"],
    }
    if qmmm_input is not None:
        # Embedding data are repeated in the golden intentionally: consumers can
        # interpret a result without reconstructing ephemeral xTB CLI files.
        golden["qmmm_input"] = qmmm_input
    return golden


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
    references = manifest.get("reference_engines", {})
    tblite_reference = references.get("tblite", {})
    xtb_reference = references.get("xtb", {})
    for engine, reference in (("tblite", tblite_reference), ("xtb", xtb_reference)):
        accuracy = reference_accuracy(reference, engine)
        templates = [reference.get("cli_command_template")]
        if engine == "xtb":
            templates.append(reference.get("qmmm_cli_command_template"))
        for template in templates:
            if not isinstance(template, list) or not any(
                template[index : index + 2] == ["--acc", accuracy]
                for index in range(max(0, len(template) - 1))
            ):
                raise ConformanceError(
                    f"manifest {engine} command template must pin --acc {accuracy}"
                )
    if xtb_reference.get("qmmm_materialization_schema") != QMMM_MATERIALIZATION_SCHEMA:
        raise ConformanceError(
            "manifest has an unsupported QMMM materialization schema"
        )
    point_hardness = xtb_reference["point_charge_hardness_hartree"]
    for case in cases:
        backends = case.get("xtbloom_backends", ["cpu", "cuda"])
        if (
            not isinstance(backends, list)
            or not backends
            or any(backend not in ("cpu", "cuda") for backend in backends)
            or len(set(backends)) != len(backends)
        ):
            raise ConformanceError(
                f"case {case['id']} xtbloom_backends must be a nonempty unique "
                "subset of ['cpu', 'cuda']"
            )
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
        qmmm_input = None
        if case.get("input_schema") == "qmmm-v1":
            qmmm_input = load_qmmm_input(input_path, case, point_hardness)
        elif "point_charge_count" in case:
            raise ConformanceError(
                f"case {case['id']} has point charges but no qmmm-v1 input schema"
            )
        else:
            # Keep gas-phase inputs directly consumable by the public C API
            # harness, not merely hash-valid for the reference executables.
            load_turbomole_coord(input_path, case)
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
        if qmmm_input is not None and golden.get("qmmm_input") != qmmm_input:
            raise ConformanceError(
                f"golden {golden_path} does not embed its exact QMMM input"
            )
        golden_properties = golden.get("properties")
        if not isinstance(golden_properties, dict):
            raise ConformanceError(f"golden {golden_path} has no properties object")
        oracle_properties = case.get("xtbloom_oracle_properties")
        if oracle_properties is not None:
            if (
                not isinstance(oracle_properties, list)
                or not oracle_properties
                or any(
                    not isinstance(property_name, str)
                    or property_name not in golden_properties
                    for property_name in oracle_properties
                )
                or len(set(oracle_properties)) != len(oracle_properties)
            ):
                raise ConformanceError(
                    f"case {case['id']} xtbloom_oracle_properties must be a "
                    "nonempty unique subset of its golden properties"
                )
            if (
                "forces_hartree_per_bohr" in golden_properties
                and "forces_hartree_per_bohr" not in oracle_properties
                and case.get("xtbloom_force_evidence")
                != "public_energy_finite_difference"
            ):
                raise ConformanceError(
                    f"case {case['id']} excludes the golden force but does not "
                    "name public_energy_finite_difference evidence"
                )
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
        expected_output_hash = case.get("reference_output_sha256")
        if not isinstance(expected_output_hash, str):
            raise ConformanceError(
                f"case {case['id']} must pin a live reference output hash"
            )
        if provenance.get("source_output_sha256") != expected_output_hash:
            raise ConformanceError(
                f"golden {golden_path} has the wrong reference output hash"
            )
        reference = references[reference_engine]
        expected_accuracy = reference_accuracy(reference, reference_engine)
        if provenance.get("generation_mode") != "live-cli" or provenance.get(
            "accuracy"
        ) != float(expected_accuracy):
            raise ConformanceError(
                f"golden {golden_path} does not pin the primary oracle accuracy"
            )
        if reference_engine == "tblite":
            accepted_templates = accepted_tblite_command_templates(
                tblite_reference, case
            )
            if (
                provenance.get("command") not in accepted_templates
                or provenance.get("command_template") not in accepted_templates
            ):
                raise ConformanceError(
                    f"golden {golden_path} has the wrong tblite command template"
                )
            if f"tblite version {tblite_reference['version']}" not in str(
                provenance.get("executable_version", "")
            ):
                raise ConformanceError(
                    f"golden {golden_path} has the wrong tblite version"
                )
            runtime = provenance.get("runtime", {})
            if (
                runtime.get("libtblite", {}).get("sha256")
                != tblite_reference["runtime_artifacts"]["libtblite_sha256"]
            ):
                raise ConformanceError(
                    f"golden {golden_path} has the wrong libtblite hash"
                )
            environment_record = provenance.get("environment", {})
            expected_set = {
                "LC_ALL": "C",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            if environment_record.get("set") != expected_set:
                raise ConformanceError(
                    f"golden {golden_path} does not pin the tblite runtime environment"
                )
        else:
            expected_command = xtb_command(Path("{executable}"), case)
            expected_template = xtb_reference[
                "qmmm_cli_command_template"
                if qmmm_input is not None
                else "cli_command_template"
            ]
            if (
                provenance.get("command") != expected_command
                or provenance.get("command_template") != expected_template
            ):
                raise ConformanceError(
                    f"golden {golden_path} has the wrong xTB command contract"
                )
            runtime = provenance.get("runtime", {})
            expected_runtime = xtb_reference["runtime_artifacts"]
            for artifact_name, expected_key in (
                ("libxtb", "libxtb_sha256"),
                ("gfn2_parameter", "gfn2_parameter_sha256"),
            ):
                if (
                    runtime.get(artifact_name, {}).get("sha256")
                    != expected_runtime[expected_key]
                ):
                    raise ConformanceError(
                        f"golden {golden_path} has the wrong {artifact_name} hash"
                    )
            environment_record = provenance.get("environment", {})
            if (
                environment_record.get("cleared_variable_prefixes") != ["XTB"]
                or environment_record.get("set", {}).get("XTBPATH")
                != "<directory-containing-pinned-param_gfn2-xtb.txt>"
            ):
                raise ConformanceError(
                    f"golden {golden_path} does not pin the xTB parameter lookup"
                )
        if qmmm_input is not None:
            if provenance.get("qmmm_input_sha256") != case["input_sha256"]:
                raise ConformanceError(
                    f"golden {golden_path} has the wrong QMMM input hash"
                )
            if provenance.get("scientific_source") != case.get("scientific_source"):
                raise ConformanceError(
                    f"golden {golden_path} has the wrong QM/MM scientific source"
                )
            expected_materialization = qmmm_materialization_provenance(qmmm_input)
            if provenance.get("materialized_input") != expected_materialization:
                raise ConformanceError(
                    f"golden {golden_path} has stale materialized QMMM input hashes"
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
            for force, grad in zip(forces, gradient, strict=True)
        ):
            raise ConformanceError(f"golden {golden_path} violates force = -gradient")
        point_charge_count = int(case.get("point_charge_count", 0))
        if point_charge_count:
            pc_forces = properties.get("point_charge_forces_hartree_per_bohr")
            pc_gradient = properties.get("point_charge_gradient_hartree_per_bohr")
            expected_pc_components = 3 * point_charge_count
            if (
                not isinstance(pc_forces, list)
                or len(pc_forces) != expected_pc_components
                or not isinstance(pc_gradient, list)
                or len(pc_gradient) != expected_pc_components
            ):
                raise ConformanceError(
                    f"golden {golden_path} has the wrong point-charge force shape"
                )
            if any(
                not math.isfinite(float(value)) for value in [*pc_forces, *pc_gradient]
            ):
                raise ConformanceError(
                    f"golden {golden_path} has non-finite point-charge forces"
                )
            if any(
                not math.isclose(float(force), -float(grad), abs_tol=0.0)
                for force, grad in zip(pc_forces, pc_gradient, strict=True)
            ):
                raise ConformanceError(
                    f"golden {golden_path} violates point-charge force = -gradient"
                )
            # The isolated QM+PC Hamiltonian is translation invariant.  The
            # packaged xTB 6.7.1 pcgrad artifact retains about eight decimals,
            # so this gate is deliberately looser than an in-memory check.
            net_force = [
                sum(float(value) for value in forces[axis::3])
                + sum(float(value) for value in pc_forces[axis::3])
                for axis in range(3)
            ]
            if any(abs(value) > 5.0e-8 for value in net_force):
                raise ConformanceError(
                    f"golden {golden_path} violates QM+PC force conservation"
                )
        if provenance.get("source_output_sha256") != sha256_json(properties):
            raise ConformanceError(
                f"golden {golden_path} does not match its normalized "
                "reference output hash"
            )
        if reference_engine == "xtb":
            expected_shapes = {
                "partial_charges_e": (int(case["atom_count"]),),
                "atomic_dipoles_e_bohr": (int(case["atom_count"]), 3),
                "atomic_quadrupoles_e_bohr2": (int(case["atom_count"]), 6),
            }
            for property_name, shape in expected_shapes.items():
                _validate_nested_shape(
                    properties.get(property_name), shape, property_name, golden_path
                )
    print(  # noqa: T201 - CLI validation report
        f"conformance manifest OK: {len(cases)} cases"
    )


def _validate_nested_shape(
    value: object, shape: tuple[int, ...], property_name: str, path: Path
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


def discover_xtb_runtime(
    executable: Path, expected_artifacts: dict[str, Any]
) -> tuple[dict[str, Any], Path | None]:
    """Discover and hash the runtime library and GFN2 parameter file when possible."""
    runtime: dict[str, Any] = {}
    library_path: Path | None = None
    try:
        completed = subprocess.run(
            ["ldd", str(executable)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "LC_ALL": "C"},
        )
        ldd_output = completed.stdout
    except OSError:
        ldd_output = ""
    for line in ldd_output.splitlines():
        match = re.search(r"\blibxtb\.so\S*\s+=>\s+(\S+)", line)
        if match is not None:
            candidate = Path(match.group(1))
            if candidate.is_file():
                library_path = candidate.resolve()
                break
    if library_path is None:
        runtime["libxtb"] = {"discovery": "ldd", "status": "unresolved"}
    else:
        runtime["libxtb"] = {
            "discovery": "ldd",
            "filename": library_path.name,
            "sha256": sha256_file(library_path),
            "status": "resolved",
        }

    parameter_candidates = [
        executable.parent.parent / "share" / "xtb" / "param_gfn2-xtb.txt",
        executable.parent / "param_gfn2-xtb.txt",
    ]
    parameter_path = next(
        (
            candidate.resolve()
            for candidate in parameter_candidates
            if candidate.is_file()
        ),
        None,
    )
    if parameter_path is None:
        runtime["gfn2_parameter"] = {
            "discovery": "executable-prefix",
            "status": "unresolved",
        }
    else:
        runtime["gfn2_parameter"] = {
            "discovery": "executable-prefix",
            "filename": parameter_path.name,
            "selection": "XTBPATH",
            "sha256": sha256_file(parameter_path),
            "status": "resolved",
        }

    for artifact_name, expected_key in (
        ("libxtb", "libxtb_sha256"),
        ("gfn2_parameter", "gfn2_parameter_sha256"),
    ):
        actual_hash = runtime[artifact_name].get("sha256")
        expected_hash = expected_artifacts.get(expected_key)
        if actual_hash is not None and actual_hash != expected_hash:
            raise ConformanceError(
                f"pinned xTB {artifact_name} SHA-256 mismatch: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return runtime, parameter_path


def xtb_environment(
    parameter_path: Path | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Create a deterministic xTB environment and document its lookup policy."""
    environment = os.environ.copy()
    for variable in list(environment):
        if variable.startswith("XTB"):
            del environment[variable]
    fixed = {
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "OMP_STACKSIZE": "4G",
        "OPENBLAS_NUM_THREADS": "1",
    }
    environment.update(fixed)
    recorded_fixed = dict(fixed)
    if parameter_path is not None:
        environment["XTBPATH"] = str(parameter_path.parent)
        recorded_fixed["XTBPATH"] = "<directory-containing-pinned-param_gfn2-xtb.txt>"
    provenance = {
        "cleared_variable_prefixes": ["XTB"],
        "inherited_environment_boundary": (
            "Non-XTB variables are inherited; the loaded libxtb and selected GFN2 "
            "parameter file are independently hashed."
        ),
        "set": recorded_fixed,
    }
    return environment, provenance


def discover_tblite_runtime(
    executable: Path, expected_artifacts: dict[str, Any]
) -> dict[str, Any]:
    """Resolve and hash the libtblite actually selected by the CLI loader."""
    runtime: dict[str, Any] = {}
    try:
        completed = subprocess.run(
            ["ldd", str(executable)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "LC_ALL": "C"},
        )
        ldd_output = completed.stdout
    except OSError:
        ldd_output = ""
    library_path: Path | None = None
    for line in ldd_output.splitlines():
        match = re.search(r"\blibtblite\.so\S*\s+=>\s+(\S+)", line)
        if match is not None:
            candidate = Path(match.group(1))
            if candidate.is_file():
                library_path = candidate.resolve()
                break
    if library_path is None:
        runtime["libtblite"] = {"discovery": "ldd", "status": "unresolved"}
    else:
        actual_hash = sha256_file(library_path)
        expected_hash = expected_artifacts.get("libtblite_sha256")
        if actual_hash != expected_hash:
            raise ConformanceError(
                "pinned tblite libtblite SHA-256 mismatch: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        runtime["libtblite"] = {
            "discovery": "ldd",
            "filename": library_path.name,
            "sha256": actual_hash,
            "status": "resolved",
        }
    return runtime


def tblite_environment() -> tuple[dict[str, str], dict[str, Any]]:
    """Select deterministic locale and single-threaded reference execution."""
    environment = os.environ.copy()
    fixed = {
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    environment.update(fixed)
    provenance = {
        "inherited_environment_boundary": (
            "Variables other than the fixed locale and thread controls are inherited; "
            "the loaded libtblite is independently hashed."
        ),
        "set": fixed,
    }
    return environment, provenance


def tblite_command(
    executable: Path, case: dict[str, Any], input_path: Path, output_path: Path
) -> list[str]:
    """Construct the command used by tblite's own validation benchmarks.

    When ``case`` carries an ``efield`` list of three finite field components
    in atomic units (Hartree per elementary charge per bohr), the command
    gains ``--efield`` with the components converted to V/angstrom using at
    least twelve significant digits. Cases without an ``efield`` entry keep
    the unchanged legacy command.
    """
    command = [
        str(executable),
        str(input_path),
        "--no-restart",
        "--method",
        "gfn2",
        "--acc",
        PRIMARY_ORACLE_ACCURACY_TEXT,
        "--grad",
        str(output_path.with_suffix(".txt")),
    ]
    charge = int(case["molecular_charge"])
    if charge:
        command.extend(["--charge", f"{charge:+d}"])
    unpaired_electrons = int(case["unpaired_electrons"])
    if unpaired_electrons:
        command.extend(["--spin", str(unpaired_electrons)])
    efield = case.get("efield")
    if efield is not None:
        if (
            not isinstance(efield, list)
            or len(efield) != 3
            or any(not math.isfinite(float(component)) for component in efield)
        ):
            raise ConformanceError(
                f"case {case['id']} efield must be a finite three-component "
                "list in atomic units"
            )
        command.extend(
            [
                "--efield",
                ",".join(
                    f"{float(component) / ELECTRIC_FIELD_VAA_PER_AU:.12g}"
                    for component in efield
                ),
            ]
        )
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
    selected = selected_cases(manifest, case_names)
    unsupported = [case["id"] for case in selected if case.get("input_schema")]
    if case_names and unsupported:
        raise ConformanceError(
            "tblite CLI generation does not support external point charges: "
            + ", ".join(unsupported)
        )
    cases = (
        [case for case in selected if not case.get("input_schema")]
        if case_names
        else [
            case
            for case in selected
            if case.get("reference_engine", "tblite") == "tblite"
            and not case.get("input_schema")
        ]
    )
    resolved_executable = Path(shutil.which(str(executable)) or executable).resolve()
    if not resolved_executable.is_file():
        raise ConformanceError(f"tblite executable does not exist: {executable}")
    reference = manifest["reference_engines"]["tblite"]
    reference_accuracy(reference, "tblite")
    version = executable_version(resolved_executable)
    if f"tblite version {reference['version']}" not in version:
        raise ConformanceError(
            "tblite executable does not match the pinned oracle: expected "
            f"version {reference['version']}, got:\n{version}"
        )
    executable_digest = sha256_file(resolved_executable)
    runtime = discover_tblite_runtime(
        resolved_executable, reference["runtime_artifacts"]
    )
    environment, environment_provenance = tblite_environment()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="xtbloom-conformance-") as temporary:
        work = Path(temporary)
        for case in cases:
            input_path = resolve_manifest_path(manifest_path, case["input"])
            raw_output = work / f"{case['id']}.json"
            command = tblite_command(resolved_executable, case, input_path, raw_output)
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
                    f"tblite failed for {case['id']} with status "
                    f"{completed.returncode}:\n"
                    f"{completed.stdout}"
                )
            raw = load_json(raw_output)
            properties = normalize_tblite_output(raw, case)
            provenance = {
                # Store the reviewed template rather than temporary absolute
                # paths, which would make otherwise identical runs differ.
                "accuracy": PRIMARY_ORACLE_ACCURACY,
                "command": reference["cli_command_template"],
                "command_template": reference["cli_command_template"],
                "engine": "tblite",
                "environment": environment_provenance,
                "executable_sha256": executable_digest,
                "executable_version": version,
                "generation_mode": "live-cli",
                "input": case["input"],
                "runtime": runtime,
                "source_output_sha256": sha256_json(properties),
                "source_revision": reference["revision"],
            }
            destination = output_dir / f"{case['id']}.json"
            dump_json(destination, canonical_golden(manifest, case, raw, provenance))
            print(f"generated {destination}")  # noqa: T201 - CLI progress output


def xtb_command(executable: Path, case: dict[str, Any]) -> list[str]:
    """Construct a deterministic single-threaded GFN2 gradient calculation."""
    command = [
        str(executable),
        "coord",
        "--gfn",
        "2",
        "--acc",
        PRIMARY_ORACLE_ACCURACY_TEXT,
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
    if case.get("input_schema") == "qmmm-v1":
        command.extend(["--input", "xcontrol"])
    return command


def generate_with_xtb(
    manifest_path: Path,
    executable: Path,
    output_dir: Path,
    case_names: list[str] | None,
) -> None:
    """Run xtb and emit normalized energy, force, and SCC-state goldens."""
    manifest = load_json(manifest_path)
    selected = selected_cases(manifest, case_names)
    cases = (
        selected
        if case_names
        else [
            case for case in selected if case.get("reference_engine", "tblite") == "xtb"
        ]
    )
    resolved_executable = Path(shutil.which(str(executable)) or executable).resolve()
    if not resolved_executable.is_file():
        raise ConformanceError(f"xtb executable does not exist: {executable}")
    version = executable_version(resolved_executable)
    executable_digest = sha256_file(resolved_executable)
    xtb_reference = manifest["reference_engines"]["xtb"]
    reference_accuracy(xtb_reference, "xtb")
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
    runtime, parameter_path = discover_xtb_runtime(
        resolved_executable, xtb_reference["runtime_artifacts"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explicit one-thread settings avoid BLAS/OpenMP scheduling differences.
    # XTB-prefixed variables are cleared before selecting the hashed parameter
    # directory, preventing a caller's XTBPATH from silently changing the model.
    environment, environment_provenance = xtb_environment(parameter_path)
    with tempfile.TemporaryDirectory(prefix="xtbloom-conformance-xtb-") as temporary:
        root = Path(temporary)
        for case in cases:
            work = root / case["id"]
            work.mkdir()
            input_path = resolve_manifest_path(manifest_path, case["input"])
            qmmm_input = None
            if case.get("input_schema") == "qmmm-v1":
                qmmm_input = load_qmmm_input(
                    input_path,
                    case,
                    xtb_reference["point_charge_hardness_hartree"],
                )
                write_xtb_qmmm_files(work, qmmm_input)
            else:
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
            pcgradient_text = None
            if qmmm_input is not None:
                try:
                    pcgradient_text = (work / "pcgrad").read_text(encoding="utf-8")
                except OSError as exc:
                    raise ConformanceError(
                        f"cannot read xtb point-charge gradient for {case['id']}: {exc}"
                    ) from exc
            properties = normalize_xtb_output(raw, gradient_text, case, pcgradient_text)
            command_template = xtb_reference[
                "qmmm_cli_command_template"
                if qmmm_input is not None
                else "cli_command_template"
            ]
            provenance = {
                "accuracy": PRIMARY_ORACLE_ACCURACY,
                "command": ["{executable}", *command[1:]],
                "command_template": command_template,
                "engine": "xtb",
                "environment": environment_provenance,
                "executable_sha256": executable_digest,
                "executable_version": version,
                "generation_mode": "live-cli",
                "input": case["input"],
                # Hash parsed scientific output, excluding xtb's absolute
                # executable path and other non-reproducible JSON metadata.
                "source_output_sha256": sha256_json(properties),
                "source_revision": xtb_reference["revision"],
                "runtime": runtime,
            }
            if qmmm_input is not None:
                provenance["materialized_input"] = qmmm_materialization_provenance(
                    qmmm_input
                )
                provenance["qmmm_input_sha256"] = sha256_file(input_path)
                provenance["scientific_source"] = case.get("scientific_source")
            destination = output_dir / f"{case['id']}.json"
            dump_json(
                destination,
                canonical_xtb_golden(
                    manifest, case, properties, provenance, qmmm_input
                ),
            )
            print(f"generated {destination}")  # noqa: T201 - CLI progress output


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
            f"tblite revision mismatch: expected {expected_revision}, "
            f"got {actual_revision}; "
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
        print(f"imported {case['id']}")  # noqa: T201 - CLI progress output
    print(  # noqa: T201 - CLI completion output
        "snapshot imported; update manifest hashes before running 'check'"
    )


def actual_properties(raw: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Accept canonical golden, raw tblite, or a minimal xtbloom result object."""
    if isinstance(raw.get("properties"), dict):
        return raw["properties"]
    if "energy" in raw and "gradient" in raw:
        return normalize_tblite_output(raw, case)
    energy = raw.get("energy_hartree", raw.get("energy"))
    forces = raw.get("forces_hartree_per_bohr", raw.get("forces"))
    properties: dict[str, Any] = {}
    if energy is not None:
        properties["energy_hartree"] = float(energy)
    if forces is not None:
        properties["forces_hartree_per_bohr"] = [float(value) for value in forces]
    if not properties:
        raise ConformanceError(
            "actual JSON must be canonical golden, raw tblite output, or contain "
            "at least one xtbloom energy/force property"
        )
    for property_name in (
        "partial_charges_e",
        "atomic_dipoles_e_bohr",
        "atomic_quadrupoles_e_bohr2",
        "point_charge_forces_hartree_per_bohr",
    ):
        if property_name in raw:
            properties[property_name] = raw[property_name]
    return properties


def compare_values(
    case_id: str,
    property_name: str,
    expected: object,
    actual: object,
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    """Compare a scalar or nested numeric array and report the worst component."""
    expected_values = _flatten_numeric(expected, property_name)
    actual_values = _flatten_numeric(actual, property_name)
    if len(expected_values) != len(actual_values):
        return (
            False,
            f"{case_id} {property_name}: shape {len(actual_values)} != "
            f"{len(expected_values)}",
        )
    worst_index = 0
    worst_error = -1.0
    worst_limit = 0.0
    passed = True
    for index, (expected_value, actual_value) in enumerate(
        zip(expected_values, actual_values, strict=True)
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


def _flatten_numeric(value: object, property_name: str) -> list[float]:
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
        minimal_xtbloom_result = (
            actual_engine is None
            and not isinstance(actual_document.get("properties"), dict)
            and not ("energy" in actual_document and "gradient" in actual_document)
        )
        oracle_properties = (
            case.get("xtbloom_oracle_properties")
            if actual_engine == "xtbloom" or minimal_xtbloom_result
            else None
        )
        tolerances = manifest["tolerances"]
        if (
            actual_engine in manifest["reference_engines"]
            and expected_engine in manifest["reference_engines"]
            and actual_engine != expected_engine
        ):
            # xtb and tblite are independent implementations with small force
            # differences. This looser gate applies only when both documents
            # explicitly identify distinct reference engines; xtbloom results
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
                ("point_charge_forces_hartree_per_bohr", "point_charge_forces"),
            )
            if property_name in expected
        )
        for property_name, tolerance_name in compared_properties:
            if oracle_properties is not None and property_name not in oracle_properties:
                print(  # noqa: T201 - CLI validation report
                    f"INFO {case['id']} {property_name}: pinned reference is "
                    "diagnostic-only for xtbloom"
                )
                continue
            if property_name not in actual:
                failures.append(f"{case['id']} is missing {property_name}")
                print(  # noqa: T201 - CLI validation report
                    f"FAIL {failures[-1]}"
                )
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
            print(  # noqa: T201 - CLI validation report
                ("PASS " if passed else "FAIL ") + message
            )
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
        print(  # noqa: T201 - CLI diagnostics
            f"conformance error: {exc}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
