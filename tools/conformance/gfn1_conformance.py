#!/usr/bin/env python3
"""Generate and verify the independent GFN1-xTB reference corpus.

This tool is deliberately separate from the production GFN2 public-API
runner: GFN1 remains unavailable in xTBloom.  It records live tblite/xTB
results only, with forces normalized as the negative Cartesian gradient.
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
    from collections.abc import Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/conformance/gfn1/manifest.json"
DEFAULT_TEMPLATE = REPOSITORY_ROOT / "data/conformance/gfn1/manifest.template.json"
ACCURACY = 1.0e-4
ACCURACY_TEXT = "0.0001"
MAX_PRIMARY_ATOL = ACCURACY
ORACLE_TIMEOUT_SECONDS = 300
QMMM_SCHEMA = "xtbloom-gfn1-xtb-pcem-cli-v1"
QMMM_INPUT_UNITS = {
    "point_charge_gammas": "hartree",
    "point_charge_positions": "bohr",
    "point_charges": "elementary_charge",
    "qm_positions": "bohr",
}
GOLDEN_UNITS = {
    "coordinates": "bohr",
    "energy": "hartree",
    "forces": "hartree/bohr",
    "gradient": "hartree/bohr",
    "molecular_charge": "elementary_charge",
}
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
TOLERANCE_UNITS = {
    "charges": "elementary_charge",
    "energy": "hartree",
    "forces": "hartree/bohr",
    "point_charge_forces": "hartree/bohr",
}
XTB_GFN1_TEST_SOURCE = {
    "git_blob_sha1": "37972be0d5a47e1e1362a392eb251d39dd0a4a74",
    "path": "test/unit/test_gfn1.f90",
    "sha256": "bc88003688e69f78139721343c71c025fe516121e2af37155c13605f7b49dd99",
}


class ConformanceError(RuntimeError):
    """An actionable GFN1 corpus, oracle, or comparison failure."""


def canonical_json(value: object) -> str:
    """Return the repository's deterministic, review-friendly JSON format."""
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object while retaining the failing path in diagnostics."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConformanceError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConformanceError(f"expected a JSON object in {path}")
    return value


def finite_number(value: object, name: str) -> float:
    """Return one exact JSON number while rejecting booleans and coercions."""
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ConformanceError(f"{name} must be a finite JSON number")
    return float(value)


def finite_vector(value: object, count: int, name: str) -> list[float]:
    """Validate a flat finite numeric vector with one exact expected extent."""
    if not isinstance(value, list) or len(value) != count:
        raise ConformanceError(f"{name} must contain exactly {count} values")
    return [finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def exact_integer(value: object, name: str) -> int:
    """Return one exact JSON integer, excluding the bool subclass."""
    if type(value) is not int:
        raise ConformanceError(f"{name} must be a JSON integer")
    return value


def validate_scientific_source(value: object, section: str, name: str) -> None:
    """Bind a derived fixture to one exact upstream source blob and section."""
    expected = {**XTB_GFN1_TEST_SOURCE, "section": section}
    if value != expected:
        raise ConformanceError(f"{name} must equal the pinned xTB GFN1 source record")


def write_json(path: Path, value: object) -> None:
    """Write canonical JSON, creating only the requested parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for exact bytes."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading large future corpora into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    """Hash normalized scientific output independent of JSON indentation."""
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def git_blob_sha1(content: bytes) -> str:
    """Hash retained bytes exactly as a loose Git blob object."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def resolve_path(relative: str) -> Path:
    """Resolve corpus paths from the repository root."""
    path = Path(relative)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def selected_cases(
    manifest: Mapping[str, Any], names: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Return selected cases while rejecting duplicate IDs and misspellings."""
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ConformanceError("manifest cases must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ConformanceError("each case must be an object with a string id")
        if case["id"] in by_id:
            raise ConformanceError(f"duplicate case ID: {case['id']}")
        by_id[case["id"]] = case
    if not names:
        return list(by_id.values())
    unknown = sorted(set(names) - set(by_id))
    if unknown:
        raise ConformanceError(f"unknown case ID(s): {', '.join(unknown)}")
    return [by_id[name] for name in names]


def load_coord(path: Path, atom_count: int) -> tuple[list[str], list[list[float]]]:
    """Read the atomic-unit Turbomole coordinate subset used by the corpus."""
    symbols: list[str] = []
    positions: list[list[float]] = []
    in_coord = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not in_coord:
            if stripped.lower() == "$coord":
                in_coord = True
            continue
        if stripped.startswith("$"):
            break
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 4:
            raise ConformanceError(f"{path}:{line_number} must be x y z symbol")
        try:
            position = [float(value) for value in fields[:3]]
        except ValueError as exc:
            raise ConformanceError(
                f"{path}:{line_number} has invalid coordinates"
            ) from exc
        if any(not math.isfinite(value) for value in position):
            raise ConformanceError(f"{path}:{line_number} has non-finite coordinates")
        positions.append(position)
        symbols.append(fields[3].capitalize())
    if len(symbols) != atom_count:
        raise ConformanceError(
            f"{path} contains {len(symbols)} atoms; expected {atom_count}"
        )
    return symbols, positions


def load_qmmm(
    path: Path, case: Mapping[str, Any], hardness: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the GFN1-specific QM plus explicit-point-charge document."""
    document = load_json(path)
    if document.get("schema_version") != 1 or document.get("method") != "GFN1-xTB":
        raise ConformanceError(f"{path} must be a schema-v1 GFN1-xTB QMMM input")
    if document.get("case_id") != case["id"]:
        raise ConformanceError(f"{path} has the wrong case_id")
    if document.get("units") != QMMM_INPUT_UNITS:
        raise ConformanceError(f"{path} has inconsistent units")
    qm = document.get("qm")
    point = document.get("external_point_charges")
    if not isinstance(qm, dict) or not isinstance(point, dict):
        raise ConformanceError(f"{path} lacks QM or point-charge data")
    atoms = int(case["atom_count"])
    points = int(case["point_charge_count"])
    _numeric_array(qm.get("positions_bohr"), (atoms, 3), "qm.positions_bohr", path)
    atomic_numbers = _atomic_numbers(qm.get("atomic_numbers"), atoms, path)
    symbols = qm.get("symbols")
    if (
        not isinstance(symbols, list)
        or len(symbols) != atoms
        or any(type(symbol) is not str for symbol in symbols)
    ):
        raise ConformanceError(f"{path} has invalid qm.symbols")
    expected_symbols = [ELEMENT_SYMBOLS[number] for number in atomic_numbers]
    if symbols != expected_symbols:
        raise ConformanceError(
            f"{path} QM symbols do not match atomic_numbers: "
            f"expected {expected_symbols}, got {symbols}"
        )
    _numeric_array(
        point.get("positions_bohr"),
        (points, 3),
        "point.positions_bohr",
        path,
    )
    for key in ("charges_e", "gammas_hartree"):
        _numeric_array(point.get(key), (points,), f"point.{key}", path)
    actual = [float(value) for value in point["gammas_hartree"]]
    if point.get("gamma_mode") == "element_hardness":
        source_numbers = point.get("source_atomic_numbers")
        source_numbers = _atomic_numbers(source_numbers, points, path)
        try:
            expected = [float(hardness[str(number)]) for number in source_numbers]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConformanceError(
                f"{path} has unsupported point-charge source elements"
            ) from exc
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(actual, expected, strict=True)
        ):
            raise ConformanceError(f"{path} point hardnesses do not match GFN1 values")
    elif point.get("gamma_mode") == "explicit":
        if "source_atomic_numbers" in point:
            raise ConformanceError(
                f"{path} explicit gamma mode must not name source elements"
            )
        if any(value <= 0.0 for value in actual):
            raise ConformanceError(f"{path} contains a non-positive explicit gamma")
    else:
        raise ConformanceError(f"{path} has an unsupported point-charge gamma mode")
    if (
        type(qm.get("molecular_charge")) is not int
        or type(qm.get("unpaired_electrons")) is not int
    ):
        raise ConformanceError(f"{path} charge and spin must be integers")
    if (
        qm["molecular_charge"] != case["molecular_charge"]
        or qm["unpaired_electrons"] != case["unpaired_electrons"]
    ):
        raise ConformanceError(f"{path} has inconsistent charge or spin")
    return document


def _numeric_array(
    value: object, shape: tuple[int, ...], name: str, path: Path
) -> None:
    """Reject wrong shapes, implicit string coercions, booleans, and non-finites."""
    if not shape:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ConformanceError(f"{path} has invalid numeric value in {name}")
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ConformanceError(f"{path} has invalid {name} shape")
    for item in value:
        _numeric_array(item, shape[1:], name, path)


def _atomic_numbers(value: object, count: int, path: Path) -> list[int]:
    """Validate exact integral atomic numbers representable by the corpus."""
    if not isinstance(value, list) or len(value) != count:
        raise ConformanceError(f"{path} has invalid atomic-number shape")
    numbers: list[int] = []
    for number in value:
        if type(number) is not int or not 1 <= number < len(ELEMENT_SYMBOLS):
            raise ConformanceError(f"{path} has unsupported atomic number {number!r}")
        numbers.append(number)
    return numbers


def materialize_qmmm(document: Mapping[str, Any]) -> dict[str, str]:
    """Convert one validated JSON input to the exact xTB PCEM CLI files."""
    qm = document["qm"]
    coord = ["$coord"]
    for position, symbol in zip(qm["positions_bohr"], qm["symbols"], strict=True):
        coord.append(
            " ".join([*(f"{float(value):.17g}" for value in position), symbol.lower()])
        )
    coord.append("$end")
    point = document["external_point_charges"]
    pcharge = [str(len(point["charges_e"]))]
    for charge, position, gamma in zip(
        point["charges_e"],
        point["positions_bohr"],
        point["gammas_hartree"],
        strict=True,
    ):
        pcharge.append(
            " ".join(
                [
                    f"{float(charge):.17g}",
                    *(f"{float(value):.17g}" for value in position),
                    f"{float(gamma):.17g}",
                ]
            )
        )
    return {
        "coord": "\n".join(coord) + "\n",
        "pcharge": "\n".join(pcharge) + "\n",
        "xcontrol": "$embedding\n input=pcharge\n gradient=pcgrad\n$end\n",
    }


def materialization_record(document: Mapping[str, Any]) -> dict[str, Any]:
    """Record hashes for every ephemeral xTB PCEM input file."""
    return {
        "schema": QMMM_SCHEMA,
        "files_sha256": {
            name: sha256_bytes(content.encode("utf-8"))
            for name, content in materialize_qmmm(document).items()
        },
    }


def validate_tolerances(manifest: Mapping[str, Any]) -> None:
    """Require strict absolute scientific thresholds with explicit units."""
    tolerances = manifest.get("tolerances")
    if not isinstance(tolerances, dict) or set(tolerances) != set(TOLERANCE_UNITS):
        raise ConformanceError("manifest tolerance set is incomplete or unexpected")
    for name, unit in TOLERANCE_UNITS.items():
        tolerance = tolerances[name]
        if not isinstance(tolerance, dict):
            raise ConformanceError(f"manifest {name} tolerance must be an object")
        atol = finite_number(tolerance.get("atol"), f"manifest {name} atol")
        rtol = finite_number(tolerance.get("rtol"), f"manifest {name} rtol")
        justification = tolerance.get("justification")
        if not 0.0 < atol <= MAX_PRIMARY_ATOL:
            raise ConformanceError(
                f"manifest {name} atol must be in (0, {MAX_PRIMARY_ATOL:g}]"
            )
        if rtol != 0.0:
            raise ConformanceError(f"manifest {name} rtol must be exactly zero")
        if tolerance.get("unit") != unit:
            raise ConformanceError(f"manifest {name} tolerance has the wrong unit")
        if not isinstance(justification, str) or not justification.strip():
            raise ConformanceError(
                f"manifest {name} tolerance needs a nonempty justification"
            )


def validate_golden_document(
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    qmmm: Mapping[str, Any] | None,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one golden before its numbers can participate in comparison."""
    if result.get("schema_version") != manifest["golden_schema_version"]:
        raise ConformanceError(f"{name} has the wrong golden schema version")
    if (
        result.get("case_id") != case["id"]
        or result.get("method") != manifest["method"]
    ):
        raise ConformanceError(f"{name} has the wrong case or method identity")
    if result.get("units") != manifest["units"]:
        raise ConformanceError(f"{name} has inconsistent units")
    if exact_integer(
        result.get("molecular_charge"), f"{name} molecular_charge"
    ) != exact_integer(
        case.get("molecular_charge"), f"case {case['id']} molecular_charge"
    ):
        raise ConformanceError(f"{name} has inconsistent molecular charge")
    if exact_integer(
        result.get("unpaired_electrons"), f"{name} unpaired_electrons"
    ) != exact_integer(
        case.get("unpaired_electrons"), f"case {case['id']} unpaired_electrons"
    ):
        raise ConformanceError(f"{name} has inconsistent unpaired electrons")
    if qmmm is None:
        if "qmmm_input" in result:
            raise ConformanceError(f"{name} unexpectedly embeds a QMMM input")
    elif result.get("qmmm_input") != qmmm:
        raise ConformanceError(f"{name} does not embed the exact QMMM input")

    properties = result.get("properties")
    if not isinstance(properties, dict):
        raise ConformanceError(f"{name} properties must be an object")
    atoms = exact_integer(case.get("atom_count"), f"case {case['id']} atom_count")
    engine = case.get("reference_engine")
    if engine not in ("tblite", "xtb"):
        raise ConformanceError(f"case {case['id']} has unsupported reference engine")
    expected_properties = {
        "energy_hartree",
        "forces_hartree_per_bohr",
        "gradient_hartree_per_bohr",
    }
    if engine == "xtb":
        expected_properties.add("partial_charges_e")
    if qmmm is not None:
        expected_properties.update(
            {
                "point_charge_forces_hartree_per_bohr",
                "point_charge_gradient_hartree_per_bohr",
            }
        )
    if set(properties) != expected_properties:
        raise ConformanceError(f"{name} has an incomplete or unexpected property set")
    finite_number(properties["energy_hartree"], f"{name} energy_hartree")
    forces = finite_vector(
        properties["forces_hartree_per_bohr"], 3 * atoms, f"{name} forces"
    )
    gradient = finite_vector(
        properties["gradient_hartree_per_bohr"], 3 * atoms, f"{name} gradient"
    )
    if any(
        force != -component for force, component in zip(forces, gradient, strict=True)
    ):
        raise ConformanceError(f"{name} violates force = -gradient")
    if engine == "xtb":
        finite_vector(properties["partial_charges_e"], atoms, f"{name} charges")
    if qmmm is not None:
        points = exact_integer(
            case.get("point_charge_count"), f"case {case['id']} point_charge_count"
        )
        pc_forces = finite_vector(
            properties["point_charge_forces_hartree_per_bohr"],
            3 * points,
            f"{name} point-charge forces",
        )
        pc_gradient = finite_vector(
            properties["point_charge_gradient_hartree_per_bohr"],
            3 * points,
            f"{name} point-charge gradient",
        )
        if any(
            force != -component
            for force, component in zip(pc_forces, pc_gradient, strict=True)
        ):
            raise ConformanceError(f"{name} violates PC force = -gradient")

    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ConformanceError(f"{name} provenance must be an object")
    references = manifest["reference_engines"]
    reference = references[engine]
    if (
        provenance.get("engine") != engine
        or provenance.get("source_revision") != reference["revision"]
        or provenance.get("input") != case["input"]
        or finite_number(provenance.get("accuracy"), f"{name} accuracy") != ACCURACY
        or provenance.get("generation_mode") != "live-cli"
    ):
        raise ConformanceError(f"{name} has wrong oracle provenance")
    if provenance.get("source_output_sha256") != sha256_json(properties):
        raise ConformanceError(f"{name} normalized output hash mismatch")
    scientific_source = case.get("scientific_source")
    if scientific_source is None:
        if "scientific_source" in provenance:
            raise ConformanceError(f"{name} has unexpected scientific-source metadata")
    elif provenance.get("scientific_source") != scientific_source:
        raise ConformanceError(f"{name} has wrong scientific-source provenance")
    if engine == "tblite":
        if (
            provenance.get("command") != reference["cli_command_template"]
            or provenance.get("command_template") != reference["cli_command_template"]
            or provenance.get("executable_sha256")
            != reference["runtime_artifacts"]["executable_sha256"]
            or provenance.get("runtime", {}).get("libtblite", {}).get("sha256")
            != reference["runtime_artifacts"]["libtblite_sha256"]
        ):
            raise ConformanceError(f"{name} has wrong tblite runtime provenance")
    elif engine == "xtb":
        expected_command = xtb_command(Path("{executable}"), case)
        expected_template = reference[
            "qmmm_cli_command_template" if qmmm is not None else "cli_command_template"
        ]
        runtime = provenance.get("runtime", {})
        if (
            provenance.get("command") != expected_command
            or provenance.get("command_template") != expected_template
            or provenance.get("executable_sha256")
            != reference["runtime_artifacts"]["executable_sha256"]
            or runtime.get("libxtb", {}).get("sha256")
            != reference["runtime_artifacts"]["libxtb_sha256"]
            or runtime.get("gfn1_parameter", {}).get("sha256")
            != reference["runtime_artifacts"]["gfn1_parameter_sha256"]
        ):
            raise ConformanceError(f"{name} has wrong xTB runtime provenance")
    if qmmm is not None:
        input_path = resolve_path(str(case["input"]))
        if provenance.get("materialized_input") != materialization_record(
            qmmm
        ) or provenance.get("qmmm_input_sha256") != sha256_file(input_path):
            raise ConformanceError(f"{name} has stale QMMM input provenance")
    return properties, provenance


def parse_gradient(text: str, atom_count: int) -> tuple[float, list[float]]:
    """Read high-precision energy and Cartesian gradient from xTB's artifact."""
    lines = text.splitlines()
    index = next((i for i, line in enumerate(lines) if "SCF energy" in line), None)
    if index is None:
        raise ConformanceError("xTB gradient artifact has no SCF energy header")
    match = re.search(r"SCF\s+energy\s*=\s*([+\-0-9.EeDd]+)", lines[index])
    if match is None:
        raise ConformanceError("cannot parse xTB gradient energy")
    energy = finite_number(
        float(match.group(1).replace("D", "E").replace("d", "e")),
        "xTB gradient energy",
    )
    rows = lines[index + 1 + atom_count : index + 1 + 2 * atom_count]
    if len(rows) != atom_count:
        raise ConformanceError("xTB gradient artifact has the wrong row count")
    gradient: list[float] = []
    for row in rows:
        fields = row.split()
        if len(fields) < 3:
            raise ConformanceError(f"malformed xTB gradient row: {row!r}")
        gradient.extend(
            finite_number(float(value.replace("D", "E")), "xTB gradient component")
            for value in fields[:3]
        )
    return energy, gradient


def parse_pcgradient(text: str, point_count: int) -> list[float]:
    """Read the three-column xTB PC gradient text artifact."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if len(rows) != point_count or any(len(row) != 3 for row in rows):
        raise ConformanceError("xTB pcgrad artifact has the wrong shape")
    return [
        finite_number(float(value.replace("D", "E")), "xTB PC gradient component")
        for row in rows
        for value in row
    ]


def normalize_tblite(raw: Mapping[str, Any], atom_count: int) -> dict[str, Any]:
    """Normalize tblite energy/gradient output to force-bearing atomic units."""
    try:
        energy = finite_number(raw["energy"], "tblite energy")
        gradient = finite_vector(raw["gradient"], 3 * atom_count, "tblite gradient")
    except KeyError as exc:
        raise ConformanceError(
            "tblite output lacks numeric energy or gradient"
        ) from exc
    return {
        "energy_hartree": energy,
        "forces_hartree_per_bohr": [-value for value in gradient],
        "gradient_hartree_per_bohr": gradient,
    }


def normalize_xtb(
    raw: Mapping[str, Any],
    gradient_text: str,
    case: Mapping[str, Any],
    pc_text: str | None,
) -> dict[str, Any]:
    """Normalize xTB energy, forces, atomic charges, and optional PC forces."""
    atoms = int(case["atom_count"])
    energy, gradient = parse_gradient(gradient_text, atoms)
    try:
        charges = finite_vector(raw["partial charges"], atoms, "xTB partial charges")
        unpaired = exact_integer(
            raw["number of unpaired electrons"], "xTB number of unpaired electrons"
        )
    except KeyError as exc:
        raise ConformanceError("xTB JSON lacks charges or spin metadata") from exc
    if unpaired != int(case["unpaired_electrons"]):
        raise ConformanceError("xTB JSON has inconsistent charge or spin shape")
    properties: dict[str, Any] = {
        "energy_hartree": energy,
        "forces_hartree_per_bohr": [-value for value in gradient],
        "gradient_hartree_per_bohr": gradient,
        "partial_charges_e": charges,
    }
    points = int(case.get("point_charge_count", 0))
    if points:
        if pc_text is None:
            raise ConformanceError("xTB produced no pcgrad artifact")
        pc_gradient = parse_pcgradient(pc_text, points)
        properties["point_charge_gradient_hartree_per_bohr"] = pc_gradient
        properties["point_charge_forces_hartree_per_bohr"] = [
            -value for value in pc_gradient
        ]
    return properties


def reference_environment(
    parameter: Path | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return the deterministic locale/thread environment and its public record."""
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("XTB"):
            del environment[name]
    fixed = {
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "OMP_STACKSIZE": "4G",
        "OPENBLAS_NUM_THREADS": "1",
    }
    environment.update(fixed)
    recorded = dict(fixed)
    if parameter is not None:
        environment["XTBPATH"] = str(parameter.parent)
        recorded["XTBPATH"] = "<directory-containing-pinned-param_gfn1-xtb.txt>"
    return environment, {
        "cleared_variable_prefixes": ["XTB"],
        "set": recorded,
        "inherited_environment_boundary": (
            "Non-XTB variables are inherited; executable, shared-library, parameter, "
            "input, and normalized-output hashes remain pinned."
        ),
    }


def ldd_library(executable: Path, pattern: str, engine: str) -> Path | None:
    """Resolve one shared object from ldd output under the active loader environment."""
    try:
        completed = subprocess.run(
            ["ldd", str(executable)],
            check=False,
            text=True,
            capture_output=True,
            env={**os.environ, "LC_ALL": "C"},
            timeout=ORACLE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConformanceError(f"{engine} shared-library discovery timed out") from exc
    for line in completed.stdout.splitlines():
        match = re.search(pattern + r"\s+=>\s+(\S+)", line)
        if match and Path(match.group(1)).is_file():
            return Path(match.group(1)).resolve()
    return None


def verify_executable(
    executable: Path, reference: Mapping[str, Any], engine: str
) -> tuple[Path, str]:
    """Resolve and verify an exact pinned oracle executable."""
    resolved = Path(shutil.which(str(executable)) or executable).resolve()
    if not resolved.is_file():
        raise ConformanceError(f"{engine} executable does not exist: {executable}")
    digest = sha256_file(resolved)
    if digest != reference["runtime_artifacts"]["executable_sha256"]:
        raise ConformanceError(f"{engine} executable SHA-256 mismatch")
    try:
        completed = subprocess.run(
            [str(resolved), "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=ORACLE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConformanceError(f"{engine} executable version check timed out") from exc
    version = completed.stdout.strip()
    expected = str(reference["version"])
    if engine == "tblite":
        valid = f"tblite version {expected}" in version
    else:
        valid = (
            f"xtb version {expected}" in version
            and str(reference["revision"])[:7] in version
        )
    if not valid:
        raise ConformanceError(f"{engine} executable has the wrong version:\n{version}")
    return resolved, version


def tblite_command(
    executable: Path, case: Mapping[str, Any], output: Path
) -> list[str]:
    """Construct the exact reviewed GFN1 tblite CLI invocation."""
    command = [
        str(executable),
        str(resolve_path(str(case["input"]))),
        "--no-restart",
        "--method",
        "gfn1",
        "--acc",
        ACCURACY_TEXT,
        "--grad",
        str(output.with_suffix(".txt")),
    ]
    if int(case["molecular_charge"]):
        command.extend(["--charge", f"{int(case['molecular_charge']):+d}"])
    command.extend(["--json", str(output)])
    return command


def xtb_command(executable: Path, case: Mapping[str, Any]) -> list[str]:
    """Construct the exact reviewed GFN1 xTB gradient invocation."""
    command = [
        str(executable),
        "coord",
        "--gfn",
        "1",
        "--acc",
        ACCURACY_TEXT,
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


def golden(
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
    properties: Mapping[str, Any],
    provenance: Mapping[str, Any],
    qmmm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical GFN1 reference result."""
    result: dict[str, Any] = {
        "case_id": case["id"],
        "method": manifest["method"],
        "molecular_charge": case["molecular_charge"],
        "provenance": provenance,
        "properties": properties,
        "schema_version": manifest["golden_schema_version"],
        "units": manifest["units"],
        "unpaired_electrons": case["unpaired_electrons"],
    }
    if qmmm is not None:
        result["qmmm_input"] = qmmm
    return result


def generate_tblite(
    manifest_path: Path, executable: Path, output_dir: Path, names: Sequence[str] | None
) -> None:
    """Generate tblite-primary closed-shell GFN1 goldens."""
    manifest = load_json(manifest_path)
    cases = [
        case
        for case in selected_cases(manifest, names)
        if case["reference_engine"] == "tblite"
    ]
    reference = manifest["reference_engines"]["tblite"]
    executable, version = verify_executable(executable, reference, "tblite")
    library = ldd_library(executable, r"\blibtblite\.so\S*", "tblite")
    if (
        library is None
        or sha256_file(library) != reference["runtime_artifacts"]["libtblite_sha256"]
    ):
        raise ConformanceError(
            "tblite libtblite could not be resolved with the pinned hash"
        )
    environment, environment_record = reference_environment()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-tblite-") as temporary:
        work = Path(temporary)
        for case in cases:
            raw_path = work / f"{case['id']}.json"
            command = tblite_command(executable, case, raw_path)
            try:
                completed = subprocess.run(
                    command,
                    cwd=work,
                    env=environment,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=ORACLE_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise ConformanceError(f"tblite timed out for {case['id']}") from exc
            if completed.returncode:
                diagnostic = f"{completed.stdout}\n{completed.stderr}"
                raise ConformanceError(f"tblite failed for {case['id']}:\n{diagnostic}")
            properties = normalize_tblite(load_json(raw_path), int(case["atom_count"]))
            provenance = {
                "accuracy": ACCURACY,
                "command": reference["cli_command_template"],
                "command_template": reference["cli_command_template"],
                "engine": "tblite",
                "environment": environment_record,
                "executable_sha256": sha256_file(executable),
                "executable_version": version,
                "generation_mode": "live-cli",
                "input": case["input"],
                "runtime": {
                    "libtblite": {
                        "filename": library.name,
                        "sha256": sha256_file(library),
                    }
                },
                "source_output_sha256": sha256_json(properties),
                "source_revision": reference["revision"],
            }
            write_json(
                output_dir / f"{case['id']}.json",
                golden(manifest, case, properties, provenance),
            )


def find_gfn1_parameter(executable: Path, explicit: Path | None) -> Path:
    """Resolve the exact GFN1 parameter file rather than trusting ambient XTBPATH."""
    if explicit is not None and explicit.name != "param_gfn1-xtb.txt":
        raise ConformanceError(
            "--parameter-file must be named exactly param_gfn1-xtb.txt because xTB "
            "selects that fixed basename from XTBPATH"
        )
    candidates = [explicit] if explicit is not None else []
    candidates.extend(
        [
            executable.parent.parent / "share/xtb/param_gfn1-xtb.txt",
            executable.parent / "param_gfn1-xtb.txt",
        ]
    )
    path = next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate is not None and candidate.is_file()
        ),
        None,
    )
    if path is None:
        raise ConformanceError(
            "cannot resolve param_gfn1-xtb.txt; provide --parameter-file"
        )
    return path


def generate_xtb(
    manifest_path: Path,
    executable: Path,
    parameter_file: Path | None,
    output_dir: Path,
    names: Sequence[str] | None,
) -> None:
    """Generate xTB-primary open-shell, PCEM, and halogen GFN1 goldens."""
    manifest = load_json(manifest_path)
    cases = [
        case
        for case in selected_cases(manifest, names)
        if case["reference_engine"] == "xtb"
    ]
    reference = manifest["reference_engines"]["xtb"]
    executable, version = verify_executable(executable, reference, "xtb")
    library = ldd_library(executable, r"\blibxtb\.so\S*", "xtb")
    if (
        library is None
        or sha256_file(library) != reference["runtime_artifacts"]["libxtb_sha256"]
    ):
        raise ConformanceError("xTB libxtb could not be resolved with the pinned hash")
    parameter = find_gfn1_parameter(executable, parameter_file)
    if (
        sha256_file(parameter)
        != reference["runtime_artifacts"]["gfn1_parameter_sha256"]
    ):
        raise ConformanceError("xTB GFN1 parameter SHA-256 mismatch")
    environment, environment_record = reference_environment(parameter)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-xtb-") as temporary:
        root = Path(temporary)
        for case in cases:
            work = root / case["id"]
            work.mkdir()
            input_path = resolve_path(str(case["input"]))
            qmmm = None
            if case.get("input_schema") == "qmmm-v1":
                qmmm = load_qmmm(
                    input_path, case, reference["point_charge_hardness_hartree"]
                )
                for name, content in materialize_qmmm(qmmm).items():
                    (work / name).write_text(content, encoding="utf-8")
            else:
                shutil.copyfile(input_path, work / "coord")
            command = xtb_command(executable, case)
            try:
                completed = subprocess.run(
                    command,
                    cwd=work,
                    env=environment,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=ORACLE_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise ConformanceError(f"xTB timed out for {case['id']}") from exc
            if completed.returncode:
                diagnostic = f"{completed.stdout}\n{completed.stderr}"
                raise ConformanceError(f"xTB failed for {case['id']}:\n{diagnostic}")
            raw = load_json(work / "xtbout.json")
            properties = normalize_xtb(
                raw,
                (work / "gradient").read_text(encoding="utf-8"),
                case,
                (work / "pcgrad").read_text(encoding="utf-8") if qmmm else None,
            )
            template = reference[
                "qmmm_cli_command_template" if qmmm else "cli_command_template"
            ]
            provenance: dict[str, Any] = {
                "accuracy": ACCURACY,
                "command": ["{executable}", *command[1:]],
                "command_template": template,
                "engine": "xtb",
                "environment": environment_record,
                "executable_sha256": sha256_file(executable),
                "executable_version": version,
                "generation_mode": "live-cli",
                "input": case["input"],
                "runtime": {
                    "gfn1_parameter": {
                        "filename": parameter.name,
                        "selection": "XTBPATH",
                        "sha256": sha256_file(parameter),
                    },
                    "libxtb": {
                        "filename": library.name,
                        "sha256": sha256_file(library),
                    },
                },
                "source_output_sha256": sha256_json(properties),
                "source_revision": reference["revision"],
            }
            scientific_source = case.get("scientific_source")
            if scientific_source is not None:
                provenance["scientific_source"] = scientific_source
            if qmmm:
                provenance.update(
                    {
                        "materialized_input": materialization_record(qmmm),
                        "qmmm_input_sha256": sha256_file(input_path),
                    }
                )
            write_json(
                output_dir / f"{case['id']}.json",
                golden(manifest, case, properties, provenance, qmmm),
            )


def check_manifest(manifest_path: Path) -> None:
    """Verify schema, provenance contracts, hashes, shapes, signs, and tolerances."""
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("golden_schema_version") != 1
    ):
        raise ConformanceError(
            "only GFN1 manifest/golden schema version 1 is supported"
        )
    if manifest.get("method") != "GFN1-xTB":
        raise ConformanceError("GFN1 corpus manifest must use method GFN1-xTB")
    if manifest.get("units") != GOLDEN_UNITS:
        raise ConformanceError("GFN1 corpus manifest has inconsistent units")
    validate_tolerances(manifest)
    references = manifest.get("reference_engines", {})
    for engine in ("tblite", "xtb"):
        reference = references.get(engine, {})
        if finite_number(reference.get("accuracy"), f"{engine} accuracy") != ACCURACY:
            raise ConformanceError(f"{engine} accuracy must be {ACCURACY_TEXT}")
        runtime = reference.get("runtime_artifacts", {})
        for name in (
            ("executable_sha256", "libtblite_sha256")
            if engine == "tblite"
            else ("executable_sha256", "libxtb_sha256", "gfn1_parameter_sha256")
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(runtime.get(name, ""))):
                raise ConformanceError(
                    f"{engine} runtime artifact {name} is not pinned"
                )
    if references.get("xtb", {}).get("gfn1_test_source") != XTB_GFN1_TEST_SOURCE:
        raise ConformanceError("xTB reference lacks the exact GFN1 unit-test source")
    for case in selected_cases(manifest, None):
        atoms = exact_integer(case.get("atom_count"), f"case {case['id']} atom_count")
        exact_integer(
            case.get("molecular_charge"), f"case {case['id']} molecular_charge"
        )
        exact_integer(
            case.get("unpaired_electrons"), f"case {case['id']} unpaired_electrons"
        )
        engine = case.get("reference_engine")
        if engine not in ("tblite", "xtb"):
            raise ConformanceError(
                f"case {case['id']} has unsupported reference engine"
            )
        input_path = resolve_path(case["input"])
        golden_path = resolve_path(case["golden"])
        for label, path in (("input", input_path), ("golden", golden_path)):
            if not path.is_file() or sha256_file(path) != case[f"{label}_sha256"]:
                raise ConformanceError(f"case {case['id']} {label} hash mismatch")
        if case.get("input_schema") == "qmmm-v1":
            exact_integer(
                case.get("point_charge_count"),
                f"case {case['id']} point_charge_count",
            )
            qmmm = load_qmmm(
                input_path, case, references["xtb"]["point_charge_hardness_hartree"]
            )
        else:
            qmmm = None
            load_coord(input_path, atoms)
            if case.get("upstream_validation_case"):
                declared_blob = str(case.get("upstream_input_git_blob", ""))
                if not re.fullmatch(r"[0-9a-f]{40}", declared_blob):
                    raise ConformanceError(
                        f"case {case['id']} lacks an upstream input Git blob"
                    )
                actual_blob = git_blob_sha1(input_path.read_bytes())
                if declared_blob != actual_blob:
                    raise ConformanceError(
                        f"case {case['id']} upstream input Git blob mismatch: "
                        f"declared {declared_blob}, retained bytes hash to "
                        f"{actual_blob}"
                    )
                if case.get("upstream_input_sha256") != case.get("input_sha256"):
                    raise ConformanceError(
                        f"case {case['id']} copied input differs from its upstream hash"
                    )
        scientific_source = case.get("scientific_source")
        if engine == "xtb" and case["id"] != "gfn1_oh_radical":
            expected_section = (
                "test_gfn1_pcem_api"
                if case.get("input_schema") == "qmmm-v1"
                else "test_gfn1_xb"
            )
            validate_scientific_source(
                scientific_source,
                expected_section,
                f"case {case['id']} scientific_source",
            )
        elif scientific_source is not None:
            raise ConformanceError(
                f"case {case['id']} has unexpected scientific-source metadata"
            )
        result = load_json(golden_path)
        _properties, provenance = validate_golden_document(
            manifest, case, result, qmmm, f"case {case['id']} golden"
        )
        if case.get("reference_output_sha256") != provenance.get(
            "source_output_sha256"
        ):
            raise ConformanceError(f"case {case['id']} manifest output hash mismatch")
    expected_manifest = finalized_manifest(DEFAULT_TEMPLATE)
    if manifest != expected_manifest:
        raise ConformanceError(
            "finalized manifest differs from the deterministic template and current "
            "input/golden hashes"
        )
    print(  # noqa: T201 - CLI validation report
        f"GFN1 conformance manifest OK: {len(selected_cases(manifest, None))} cases"
    )


def compare(manifest_path: Path, actual_dir: Path, names: Sequence[str] | None) -> None:
    """Compare live regenerated oracle files with committed GFN1 goldens."""
    manifest = load_json(manifest_path)
    failures: list[str] = []
    for case in selected_cases(manifest, names):
        expected = load_json(resolve_path(case["golden"]))
        input_path = resolve_path(str(case["input"]))
        qmmm = (
            load_qmmm(
                input_path,
                case,
                manifest["reference_engines"]["xtb"]["point_charge_hardness_hartree"],
            )
            if case.get("input_schema") == "qmmm-v1"
            else None
        )
        try:
            expected_properties, expected_provenance = validate_golden_document(
                manifest, case, expected, qmmm, f"case {case['id']} committed golden"
            )
        except ConformanceError as exc:
            failures.append(f"{case['id']}: invalid committed golden: {exc}")
            continue
        actual_path = actual_dir / f"{case['id']}.json"
        if not actual_path.is_file():
            failures.append(f"{case['id']}: missing {actual_path}")
            continue
        actual = load_json(actual_path)
        try:
            actual_properties, actual_provenance = validate_golden_document(
                manifest, case, actual, qmmm, f"case {case['id']} actual result"
            )
        except ConformanceError as exc:
            failures.append(f"{case['id']}: invalid actual result: {exc}")
            continue
        identity_errors = []
        expected_identity = {
            key: value
            for key, value in expected_provenance.items()
            if key != "source_output_sha256"
        }
        actual_identity = {
            key: value
            for key, value in actual_provenance.items()
            if key != "source_output_sha256"
        }
        if actual_identity != expected_identity:
            identity_errors.append(
                "provenance identity differs from the committed oracle"
            )
        if identity_errors:
            failures.append(
                f"{case['id']}: identity mismatch: " + "; ".join(identity_errors)
            )
            continue
        for property_name, tolerance_name in (
            ("energy_hartree", "energy"),
            ("forces_hartree_per_bohr", "forces"),
            ("partial_charges_e", "charges"),
            ("point_charge_forces_hartree_per_bohr", "point_charge_forces"),
        ):
            expected_value = expected_properties.get(property_name)
            if expected_value is None:
                continue
            actual_value = actual_properties.get(property_name)
            if actual_value is None:
                failures.append(f"{case['id']}: missing {property_name}")
                continue
            expected_flat = (
                [
                    finite_number(
                        expected_value, f"{case['id']} expected {property_name}"
                    )
                ]
                if not isinstance(expected_value, list)
                else _flatten(expected_value, f"{case['id']} expected {property_name}")
            )
            actual_flat = (
                [finite_number(actual_value, f"{case['id']} actual {property_name}")]
                if not isinstance(actual_value, list)
                else _flatten(actual_value, f"{case['id']} actual {property_name}")
            )
            if len(expected_flat) != len(actual_flat):
                failures.append(f"{case['id']}: {property_name} shape mismatch")
                continue
            error = max(
                (abs(a - b) for a, b in zip(actual_flat, expected_flat, strict=True)),
                default=0.0,
            )
            limit = float(manifest["tolerances"][tolerance_name]["atol"])
            if not math.isfinite(error) or error > limit:
                failures.append(
                    f"{case['id']}: {property_name} error {error:.3e} > {limit:.3e}"
                )
    if failures:
        raise ConformanceError("GFN1 comparison failed:\n" + "\n".join(failures))
    print(  # noqa: T201 - CLI validation report
        f"GFN1 comparison OK: {len(selected_cases(manifest, names))} cases"
    )


def _flatten(value: list[Any], name: str) -> list[float]:
    """Flatten nested numeric reference properties without accepting objects."""
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, list):
            result.extend(_flatten(item, f"{name}[{index}]"))
        else:
            result.append(finite_number(item, f"{name}[{index}]"))
    return result


def parser() -> argparse.ArgumentParser:
    """Build the standalone GFN1 oracle CLI."""
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "check", help="verify committed GFN1 inputs, goldens, and provenance"
    )
    for name in ("generate-tblite", "generate-xtb"):
        command = sub.add_parser(name)
        command.add_argument("--executable", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--case", action="append", dest="cases")
        if name == "generate-xtb":
            command.add_argument("--parameter-file", type=Path)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--actual-dir", type=Path, required=True)
    comparison.add_argument("--case", action="append", dest="cases")
    finalize = sub.add_parser(
        "finalize-manifest",
        help=(
            "derive input, golden, and normalized-output hashes into the committed "
            "manifest"
        ),
    )
    finalize.add_argument("--template", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return root


def finalized_manifest(template: Path) -> dict[str, Any]:
    """Return the deterministic manifest derived from reviewed artifacts."""
    manifest = load_json(template)
    for case in selected_cases(manifest, None):
        input_path = resolve_path(case["input"])
        golden_path = resolve_path(case["golden"])
        if not input_path.is_file() or not golden_path.is_file():
            raise ConformanceError(f"case {case['id']} input or golden is missing")
        result = load_json(golden_path)
        case["input_sha256"] = sha256_file(input_path)
        case["golden_sha256"] = sha256_file(golden_path)
        case["reference_output_sha256"] = result["provenance"]["source_output_sha256"]
        if case.get("upstream_validation_case"):
            case["upstream_input_sha256"] = case["input_sha256"]
    return manifest


def finalize_manifest(template: Path, output: Path) -> None:
    """Build the hash-pinned manifest from reviewed inputs and live goldens."""
    manifest = finalized_manifest(template)
    write_json(output, manifest)


def main(argv: Iterable[str] | None = None) -> int:
    """Run one deterministic GFN1 corpus operation."""
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "check":
            check_manifest(arguments.manifest)
        elif arguments.command == "generate-tblite":
            generate_tblite(
                arguments.manifest,
                arguments.executable,
                arguments.output_dir,
                arguments.cases,
            )
        elif arguments.command == "generate-xtb":
            generate_xtb(
                arguments.manifest,
                arguments.executable,
                arguments.parameter_file,
                arguments.output_dir,
                arguments.cases,
            )
        elif arguments.command == "finalize-manifest":
            finalize_manifest(arguments.template, arguments.output)
        else:
            compare(arguments.manifest, arguments.actual_dir, arguments.cases)
    except (ConformanceError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
