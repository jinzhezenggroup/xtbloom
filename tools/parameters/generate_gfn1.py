#!/usr/bin/env python3
"""Generate deterministic GFN1-xTB parameter artifacts from pinned tblite.

The tblite ``param --method gfn1`` export is the scientific source of truth.
The pinned dxtb TOML is checked only for semantic agreement and is never used
to generate xTBloom data.  Strict schema validation prevents newly introduced
upstream fields or GFN1-specific model terms from being silently discarded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import locale
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

import tomllib

_COMMON_PATH = Path(__file__).with_name("generate_gfn2.py")
_COMMON_SPEC = importlib.util.spec_from_file_location(
    "xtbloom_generate_gfn2_common", _COMMON_PATH
)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError(f"cannot load parameter helpers from {_COMMON_PATH}")
COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(COMMON)

ParameterError = COMMON.ParameterError
ELEMENT_SYMBOLS = COMMON.ELEMENT_SYMBOLS
ATOMIC_NUMBER = COMMON.ATOMIC_NUMBER
ANGULAR_MOMENTUM = COMMON.ANGULAR_MOMENTUM
SHELL_RE = COMMON.SHELL_RE
ELEMENT_SCALARS = COMMON.ELEMENT_SCALARS
ELEMENT_SHELL_ARRAYS = COMMON.ELEMENT_SHELL_ARRAYS
canonical_json_bytes = COMMON.canonical_json_bytes
sha256_bytes = COMMON.sha256_bytes
_mapping = COMMON._mapping
_expect_keys = COMMON._expect_keys
_number = COMMON._number
_integer = COMMON._integer
_boolean = COMMON._boolean
_string = COMMON._string
_cpp_float = COMMON._cpp_float
_cpp_string = COMMON._cpp_string
_run = COMMON._run
_git = COMMON._git

SCHEMA_VERSION = 1
METHOD = "gfn1-xtb"
UPSTREAM_REPOSITORY = "https://github.com/tblite/tblite"
UPSTREAM_LICENSE = "LGPL-3.0-or-later"
DXTB_REPOSITORY = "https://github.com/grimme-lab/dxtb"
RAW_FILENAME = "gfn1.toml"
JSON_FILENAME = "gfn1.json"
HEADER_FILENAME = "gfn1.hpp"
MANIFEST_FILENAME = "gfn1_manifest.json"

SOURCE_PATHS = (
    "app/driver_param.f90",
    "src/tblite/param.f90",
    "src/tblite/param",
    "src/tblite/xtb/gfn1.f90",
)
DEFAULT_DXTB_PATH = "src/dxtb/_src/param/gfn1/gfn1-xtb.toml"


def _shell_scale(shell: Mapping[str, Any], first: int, second: int) -> float:
    names = "spd"
    direct = names[first] + names[second]
    if direct in shell:
        return _number(shell[direct], f"hamiltonian.xtb.shell.{direct}")
    diagonal_first = _number(
        shell[names[first] * 2], f"hamiltonian.xtb.shell.{names[first] * 2}"
    )
    diagonal_second = _number(
        shell[names[second] * 2], f"hamiltonian.xtb.shell.{names[second] * 2}"
    )
    return 0.5 * (diagonal_first + diagonal_second)


def _normalize_hamiltonian(source: Mapping[str, Any]) -> dict[str, Any]:
    _expect_keys(source, "hamiltonian", ("xtb",))
    xtb = _mapping(source["xtb"], "hamiltonian.xtb")
    _expect_keys(
        xtb,
        "hamiltonian.xtb",
        ("wexp", "kpol", "enscale", "cn", "shell", "kpair"),
    )
    shell = _mapping(xtb["shell"], "hamiltonian.xtb.shell")
    allowed_shell_keys = {
        first + second
        for first_index, first in enumerate("spd")
        for second in "spd"[first_index:]
    }
    unknown_shell = sorted(shell.keys() - allowed_shell_keys)
    if unknown_shell:
        raise ParameterError(
            "hamiltonian.xtb.shell has unsupported fields: " + ", ".join(unknown_shell)
        )
    for diagonal in ("ss", "pp", "dd"):
        if diagonal not in shell:
            raise ParameterError(f"hamiltonian.xtb.shell is missing {diagonal}")

    shell_pair_scale = [
        {
            "angular_momenta": [first, second],
            "value": _shell_scale(shell, first, second),
        }
        for first in range(3)
        for second in range(first, 3)
    ]

    pair_table = _mapping(xtb["kpair"], "hamiltonian.xtb.kpair")
    pair_overrides = []
    seen_pairs: set[tuple[int, int]] = set()
    for key, value in pair_table.items():
        symbols = key.split("-")
        if len(symbols) != 2 or any(symbol not in ATOMIC_NUMBER for symbol in symbols):
            raise ParameterError(f"invalid element pair key: {key}")
        atomic_numbers = tuple(sorted(ATOMIC_NUMBER[symbol] for symbol in symbols))
        if atomic_numbers in seen_pairs:
            raise ParameterError(f"duplicate symmetric element pair: {key}")
        seen_pairs.add(atomic_numbers)
        scale = _number(value, f"hamiltonian.xtb.kpair.{key}")
        if scale == 1.0:
            raise ParameterError(f"redundant default pair scale in export: {key}")
        pair_overrides.append({"atomic_numbers": list(atomic_numbers), "value": scale})
    pair_overrides.sort(key=lambda item: item["atomic_numbers"])

    coordination_number = _string(xtb["cn"], "hamiltonian.xtb.cn")
    if coordination_number != "exp":
        raise ParameterError(
            f"unsupported GFN1 coordination-number model: {coordination_number}"
        )
    return {
        "wexp": _number(xtb["wexp"], "hamiltonian.xtb.wexp"),
        "kpol": _number(xtb["kpol"], "hamiltonian.xtb.kpol"),
        "enscale": _number(xtb["enscale"], "hamiltonian.xtb.enscale"),
        "coordination_number": coordination_number,
        "shell_pair_scale": shell_pair_scale,
        "pair_scale_default": 1.0,
        "pair_scale_overrides": pair_overrides,
    }


def _normalize_elements(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    _expect_keys(source, "element", ELEMENT_SYMBOLS)
    elements = []
    for atomic_number, symbol in enumerate(ELEMENT_SYMBOLS, 1):
        location = f"element.{symbol}"
        record = _mapping(source[symbol], location)
        _expect_keys(record, location, ELEMENT_SCALARS + ELEMENT_SHELL_ARRAYS)
        raw_shells = record["shells"]
        if not isinstance(raw_shells, list) or not raw_shells:
            raise ParameterError(f"{location}.shells must be a non-empty array")
        shell_count = len(raw_shells)
        if shell_count > 3:
            raise ParameterError(f"{location} has more than three GFN1 shells")
        for field in ELEMENT_SHELL_ARRAYS[1:]:
            values = record[field]
            if not isinstance(values, list) or len(values) != shell_count:
                raise ParameterError(
                    f"{location}.{field} must contain exactly {shell_count} values"
                )

        seen_momenta: set[int] = set()
        shells = []
        for shell_index, label_value in enumerate(raw_shells):
            label = _string(label_value, f"{location}.shells[{shell_index}]")
            match = SHELL_RE.fullmatch(label)
            if match is None:
                raise ParameterError(f"unsupported shell label {label!r} in {location}")
            momentum = ANGULAR_MOMENTUM[match.group(2)]
            if momentum > 2:
                raise ParameterError(f"GFN1 supports only s, p, and d shells: {label}")
            is_valence = momentum not in seen_momenta
            seen_momenta.add(momentum)
            reference_occupation = _number(
                record["refocc"][shell_index], f"{location}.refocc"
            )
            if not is_valence and reference_occupation != 0.0:
                raise ParameterError(
                    "repeated angular-momentum shell "
                    f"{location}.{label} must have zero refocc"
                )
            shells.append(
                {
                    "index": shell_index,
                    "principal_quantum_number": int(match.group(1)),
                    "angular_momentum": momentum,
                    "is_valence": is_valence,
                    "level": _number(
                        record["levels"][shell_index], f"{location}.levels"
                    ),
                    "slater": _number(
                        record["slater"][shell_index], f"{location}.slater"
                    ),
                    "ngauss": _integer(
                        record["ngauss"][shell_index], f"{location}.ngauss"
                    ),
                    "reference_occupation": reference_occupation,
                    "shell_polynomial": _number(
                        record["shpoly"][shell_index], f"{location}.shpoly"
                    ),
                    "coordination_number_scale": _number(
                        record["kcn"][shell_index], f"{location}.kcn"
                    ),
                    "shell_hubbard_scale": _number(
                        record["lgam"][shell_index], f"{location}.lgam"
                    ),
                }
            )
        element = {"atomic_number": atomic_number, "symbol": symbol, "shells": shells}
        element.update(
            {
                field: _number(record[field], f"{location}.{field}")
                for field in ELEMENT_SCALARS
            }
        )
        elements.append(element)
    return elements


def normalize_export(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a parsed tblite GFN1 TOML export."""
    source = _mapping(document, "root")
    _expect_keys(
        source,
        "root",
        (
            "meta",
            "hamiltonian",
            "dispersion",
            "repulsion",
            "halogen",
            "charge",
            "thirdorder",
            "element",
        ),
    )
    meta = _mapping(source["meta"], "meta")
    _expect_keys(meta, "meta", ("name", "reference", "version", "format"))

    dispersion_root = _mapping(source["dispersion"], "dispersion")
    _expect_keys(dispersion_root, "dispersion", ("d3",))
    dispersion = _mapping(dispersion_root["d3"], "dispersion.d3")
    _expect_keys(dispersion, "dispersion.d3", ("s6", "s8", "a1", "a2", "s9"))

    repulsion_root = _mapping(source["repulsion"], "repulsion")
    _expect_keys(repulsion_root, "repulsion", ("effective",))
    repulsion = _mapping(repulsion_root["effective"], "repulsion.effective")
    _expect_keys(repulsion, "repulsion.effective", ("kexp",), ("klight",))
    repulsion_kexp = _number(repulsion["kexp"], "repulsion.effective.kexp")
    repulsion_klight = _number(
        repulsion.get("klight", repulsion_kexp), "repulsion.effective.klight"
    )

    charge_root = _mapping(source["charge"], "charge")
    _expect_keys(charge_root, "charge", ("effective",))
    charge = _mapping(charge_root["effective"], "charge.effective")
    _expect_keys(charge, "charge.effective", ("gexp", "average"))
    charge_average = _string(charge["average"], "charge.effective.average")
    if charge_average != "harmonic":
        raise ParameterError(f"unsupported GFN1 charge average: {charge_average}")

    thirdorder_root = _mapping(source["thirdorder"], "thirdorder")
    _expect_keys(thirdorder_root, "thirdorder", ("shell",))
    thirdorder_shell = _boolean(thirdorder_root["shell"], "thirdorder.shell")
    if thirdorder_shell:
        raise ParameterError("GFN1 requires atomwise, not shellwise, third order")

    halogen_root = _mapping(source["halogen"], "halogen")
    _expect_keys(halogen_root, "halogen", ("classical",))
    halogen = _mapping(halogen_root["classical"], "halogen.classical")
    _expect_keys(halogen, "halogen.classical", ("damping", "rscale"))

    elements = _normalize_elements(_mapping(source["element"], "element"))
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "meta": {
            "name": _string(meta["name"], "meta.name"),
            "reference": _string(meta["reference"], "meta.reference"),
            "version": _integer(meta["version"], "meta.version"),
            "format": _integer(meta["format"], "meta.format"),
        },
        "hamiltonian": _normalize_hamiltonian(
            _mapping(source["hamiltonian"], "hamiltonian")
        ),
        "dispersion": {
            "model": "d3",
            "self_consistent": False,
            **{
                key: _number(dispersion[key], f"dispersion.d3.{key}")
                for key in ("s6", "s8", "a1", "a2", "s9")
            },
        },
        "repulsion": {"kexp": repulsion_kexp, "klight": repulsion_klight},
        "charge": {
            "gexp": _number(charge["gexp"], "charge.effective.gexp"),
            "average": charge_average,
        },
        "thirdorder": {"mode": "atom", "shell_resolved": thirdorder_shell},
        "halogen": {
            "damping": _number(halogen["damping"], "halogen.classical.damping"),
            "radius_scale": _number(halogen["rscale"], "halogen.classical.rscale"),
        },
        "elements": elements,
    }


def parse_and_normalize(raw_toml: bytes) -> dict[str, Any]:
    """Parse a UTF-8 tblite export and return validated GFN1 parameters."""
    try:
        document = tomllib.loads(raw_toml.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ParameterError(f"invalid UTF-8 tblite TOML export: {exc}") from exc
    return normalize_export(document)


def render_header(parameters: Mapping[str, Any], source_revision: str) -> bytes:
    """Render trivially-copyable GFN1 tables for later CPU/device consumers."""
    elements = parameters["elements"]
    shells = [shell for element in elements for shell in element["shells"]]
    hamiltonian = parameters["hamiltonian"]
    entries = {
        tuple(entry["angular_momenta"]): entry["value"]
        for entry in hamiltonian["shell_pair_scale"]
    }
    shell_matrix = [
        entries[tuple(sorted((first, second)))]
        for first in range(3)
        for second in range(3)
    ]
    lines = [
        "// Generated by tools/parameters/generate_gfn1.py; do not edit.",
        "// SPDX-License-Identifier: LGPL-3.0-or-later",
        "// Parameter source: tblite (LGPL-3.0-or-later).",
        f"// tblite revision: {source_revision}",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <type_traits>",
        "",
        "namespace xtbloom::parameters::gfn1 {",
        "",
        f"inline constexpr std::uint32_t kSchemaVersion = {SCHEMA_VERSION}u;",
        f"inline constexpr char kSourceRevision[] = {_cpp_string(source_revision)};",
        f"inline constexpr std::size_t kElementCount = {len(elements)}u;",
        f"inline constexpr std::size_t kShellCount = {len(shells)}u;",
        "",
        "struct GlobalParameters {",
        "  double hamiltonian_wexp;",
        "  double hamiltonian_kpol;",
        "  double hamiltonian_enscale;",
        "  std::array<double, 9> shell_pair_scale;",
        "  double pair_scale_default;",
        "  double dispersion_s6;",
        "  double dispersion_s8;",
        "  double dispersion_a1;",
        "  double dispersion_a2;",
        "  double dispersion_s9;",
        "  double repulsion_kexp;",
        "  double repulsion_klight;",
        "  double charge_gexp;",
        "  double halogen_damping;",
        "  double halogen_radius_scale;",
        "  std::uint8_t coordination_number_model;  // 1 = exp",
        "  std::uint8_t charge_average;  // 1 = harmonic",
        "  std::uint8_t dispersion_model;  // 1 = D3",
        "  bool thirdorder_shell_resolved;",
        "};",
        "",
        "struct ElementParameters {",
        "  std::uint16_t shell_offset;",
        "  std::uint8_t atomic_number;",
        "  std::uint8_t shell_count;",
        "  double gam;",
        "  double gam3;",
        "  double zeff;",
        "  double arep;",
        "  double xbond;",
        "  double electronegativity;",
        "  double dipole_kernel;",
        "  double quadrupole_kernel;",
        "  double multipole_radius;",
        "  double multipole_valence_cn;",
        "};",
        "",
        "struct ShellParameters {",
        "  std::uint8_t principal_quantum_number;",
        "  std::uint8_t angular_momentum;",
        "  std::uint8_t gaussian_count;",
        "  bool is_valence;",
        "  double level;",
        "  double slater;",
        "  double reference_occupation;",
        "  double shell_polynomial;",
        "  double coordination_number_scale;",
        "  double shell_hubbard_scale;",
        "};",
        "",
        "struct PairScaleOverride {",
        "  std::uint8_t first_atomic_number;",
        "  std::uint8_t second_atomic_number;",
        "  double value;",
        "};",
        "",
        "static_assert(std::is_trivially_copyable_v<GlobalParameters>);",
        "static_assert(std::is_trivially_copyable_v<ElementParameters>);",
        "static_assert(std::is_trivially_copyable_v<ShellParameters>);",
        "static_assert(std::is_trivially_copyable_v<PairScaleOverride>);",
        "",
        "inline constexpr GlobalParameters kGlobal{",
        f"  {_cpp_float(hamiltonian['wexp'])},",
        f"  {_cpp_float(hamiltonian['kpol'])},",
        f"  {_cpp_float(hamiltonian['enscale'])},",
        "  {{" + ", ".join(_cpp_float(value) for value in shell_matrix) + "}},",
        f"  {_cpp_float(hamiltonian['pair_scale_default'])},",
        *[
            f"  {_cpp_float(parameters['dispersion'][key])},"
            for key in ("s6", "s8", "a1", "a2", "s9")
        ],
        f"  {_cpp_float(parameters['repulsion']['kexp'])},",
        f"  {_cpp_float(parameters['repulsion']['klight'])},",
        f"  {_cpp_float(parameters['charge']['gexp'])},",
        f"  {_cpp_float(parameters['halogen']['damping'])},",
        f"  {_cpp_float(parameters['halogen']['radius_scale'])},",
        "  1u,",
        "  1u,",
        "  1u,",
        "  false,",
        "};",
        "",
        "inline constexpr std::array<ElementParameters, kElementCount> kElements{{",
    ]
    shell_offset = 0
    for element in elements:
        values = (
            f"{shell_offset}u",
            f"{element['atomic_number']}u",
            f"{len(element['shells'])}u",
            _cpp_float(element["gam"]),
            _cpp_float(element["gam3"]),
            _cpp_float(element["zeff"]),
            _cpp_float(element["arep"]),
            _cpp_float(element["xbond"]),
            _cpp_float(element["en"]),
            _cpp_float(element["dkernel"]),
            _cpp_float(element["qkernel"]),
            _cpp_float(element["mprad"]),
            _cpp_float(element["mpvcn"]),
        )
        lines.append("  {" + ", ".join(values) + "},")
        shell_offset += len(element["shells"])
    lines.extend(
        (
            "}};",
            "",
            "inline constexpr std::array<ShellParameters, kShellCount> kShells{{",
        )
    )
    for shell in shells:
        values = (
            f"{shell['principal_quantum_number']}u",
            f"{shell['angular_momentum']}u",
            f"{shell['ngauss']}u",
            str(shell["is_valence"]).lower(),
            _cpp_float(shell["level"]),
            _cpp_float(shell["slater"]),
            _cpp_float(shell["reference_occupation"]),
            _cpp_float(shell["shell_polynomial"]),
            _cpp_float(shell["coordination_number_scale"]),
            _cpp_float(shell["shell_hubbard_scale"]),
        )
        lines.append("  {" + ", ".join(values) + "},")
    lines.extend(("}};", ""))
    pair_overrides = hamiltonian["pair_scale_overrides"]
    lines.append(
        "inline constexpr std::array<PairScaleOverride, "
        f"{len(pair_overrides)}u> kPairScaleOverrides{{{{"
    )
    for override in pair_overrides:
        first, second = override["atomic_numbers"]
        lines.append(f"  {{{first}u, {second}u, {_cpp_float(override['value'])}}},")
    lines.extend(
        (
            "}};",
            "",
            "[[nodiscard]] constexpr const ElementParameters* find_element("
            "std::uint32_t atomic_number) noexcept {",
            "  return atomic_number >= 1u && atomic_number <= kElementCount ? "
            "&kElements[atomic_number - 1u] : nullptr;",
            "}",
            "",
            "[[nodiscard]] constexpr double pair_scale(std::uint32_t first, "
            "std::uint32_t second) noexcept {",
            "  for (const auto& value : kPairScaleOverrides) {",
            "    if ((value.first_atomic_number == first && "
            "value.second_atomic_number == second) ||",
            "        (value.first_atomic_number == second && "
            "value.second_atomic_number == first)) return value.value;",
            "  }",
            "  return kGlobal.pair_scale_default;",
            "}",
            "",
            "}  // namespace xtbloom::parameters::gfn1",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _source_provenance(source_dir: Path, revision_spec: str) -> dict[str, Any]:
    revision = _git(source_dir, "rev-parse", f"{revision_spec}^{{commit}}")
    paths = sorted(
        filter(
            None,
            _git(
                source_dir,
                "ls-tree",
                "-r",
                "--name-only",
                revision,
                "--",
                *SOURCE_PATHS,
            ).splitlines(),
        )
    )
    if not paths:
        raise ParameterError(f"no tblite GFN1 parameter sources found in {source_dir}")
    digest = hashlib.sha256()
    for path in paths:
        content = subprocess.run(
            ("git", "-C", str(source_dir), "show", f"{revision}:{path}"),
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    license_content = subprocess.run(
        ("git", "-C", str(source_dir), "show", f"{revision}:COPYING.LESSER"),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "repository": UPSTREAM_REPOSITORY,
        "revision": revision,
        "parameter_sources_sha256": digest.hexdigest(),
        "parameter_source_paths": paths,
        "license": {
            "spdx": UPSTREAM_LICENSE,
            "file": "COPYING.LESSER",
            "sha256": sha256_bytes(license_content),
        },
    }


def _export_tblite(tblite: Path) -> tuple[bytes, dict[str, str]]:
    executable = tblite.resolve(strict=True)
    version = _run((str(executable), "--version"))
    with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-") as directory:
        output = Path(directory) / RAW_FILENAME
        _run((str(executable), "param", "--method", "gfn1", "--output", str(output)))
        raw = output.read_bytes().replace(b"\r\n", b"\n")
    if not raw.endswith(b"\n"):
        raw += b"\n"
    return raw, {
        "version": version,
        "executable_sha256": sha256_bytes(executable.read_bytes()),
    }


def _semantic_cross_check(authoritative: bytes, cross_check: bytes) -> dict[str, Any]:
    """Require equal structure and values within two binary64 ULPs."""
    left = tomllib.loads(authoritative.decode("utf-8"))
    right = tomllib.loads(cross_check.decode("utf-8"))
    differing = 0
    max_ulp = 0.0

    def compare(first: object, second: object, location: str) -> None:
        nonlocal differing, max_ulp
        if isinstance(first, dict) and isinstance(second, dict):
            if set(first) != set(second):
                raise ParameterError(f"dxtb cross-check keys differ at {location}")
            for key in first:
                compare(first[key], second[key], f"{location}.{key}")
            return
        if isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second):
                raise ParameterError(f"dxtb cross-check length differs at {location}")
            for index, (one, two) in enumerate(zip(first, second, strict=True)):
                compare(one, two, f"{location}[{index}]")
            return
        if isinstance(first, float) and isinstance(second, float):
            if first != second:
                differing += 1
                spacing = max(math.ulp(first), math.ulp(second))
                distance = abs(first - second) / spacing
                max_ulp = max(max_ulp, distance)
                if distance > 2.0:
                    raise ParameterError(
                        f"dxtb cross-check differs by {distance:g} ULP at {location}"
                    )
            return
        if first != second:
            raise ParameterError(f"dxtb cross-check value differs at {location}")

    compare(left, right, "root")
    return {
        "comparison": "parsed TOML equality with binary64 tolerance",
        "maximum_ulp_difference": max_ulp,
        "differing_binary64_values": differing,
    }


def _dxtb_provenance(
    source_dir: Path, revision_spec: str, path: str, authoritative: bytes
) -> dict[str, Any]:
    revision = _git(source_dir, "rev-parse", f"{revision_spec}^{{commit}}")
    content = subprocess.run(
        ("git", "-C", str(source_dir), "show", f"{revision}:{path}"),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    result = {
        "role": "non-authoritative semantic cross-check",
        "repository": DXTB_REPOSITORY,
        "revision": revision,
        "path": path,
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }
    result.update(_semantic_cross_check(authoritative, content))
    return result


def _inspection_provenance(
    source_dir: Path, authoritative_revision: str, inspection_spec: str
) -> dict[str, Any]:
    """Record the clean owner-requested checkout without making it authoritative."""
    inspection_revision = _git(source_dir, "rev-parse", f"{inspection_spec}^{{commit}}")
    try:
        _run(
            (
                "git",
                "-C",
                str(source_dir),
                "merge-base",
                "--is-ancestor",
                authoritative_revision,
                inspection_revision,
            )
        )
    except ParameterError as exc:
        raise ParameterError(
            "authoritative tblite revision is not an ancestor of the "
            "inspection revision"
        ) from exc
    changed = sorted(
        filter(
            None,
            _git(
                source_dir,
                "diff",
                "--name-only",
                f"{authoritative_revision}..{inspection_revision}",
                "--",
                *SOURCE_PATHS,
            ).splitlines(),
        )
    )
    diff = subprocess.run(
        (
            "git",
            "-C",
            str(source_dir),
            "diff",
            "--binary",
            f"{authoritative_revision}..{inspection_revision}",
            "--",
            *SOURCE_PATHS,
        ),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "role": "owner-requested clean-checkout inspection only",
        "revision": inspection_revision,
        "authoritative_revision_is_ancestor": True,
        "changed_parameter_source_paths": changed,
        "source_diff_sha256": sha256_bytes(diff),
        "reviewed_classification": (
            "formatting/style only; the tblite 0.7.0 release revision remains "
            "the scientific source"
        ),
    }


def _generator_metadata() -> dict[str, Any]:
    return {
        "path": "tools/parameters/generate_gfn1.py",
        "sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "shared_helper": {
            "path": "tools/parameters/generate_gfn2.py",
            "sha256": sha256_bytes(_COMMON_PATH.read_bytes()),
        },
    }


def build_artifacts(raw_toml: bytes, provenance: Mapping[str, Any]) -> dict[str, bytes]:
    """Build deterministic GFN1 artifacts from an export and its provenance."""
    parameters = parse_and_normalize(raw_toml)
    source = _mapping(provenance.get("source"), "manifest.source")
    revision = _string(source.get("revision"), "manifest.source.revision")
    normalized = canonical_json_bytes(parameters)
    header = render_header(parameters, revision)
    manifest = copy.deepcopy(dict(provenance))
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "generator": _generator_metadata(),
            "outputs": {
                RAW_FILENAME: {
                    "bytes": len(raw_toml),
                    "sha256": sha256_bytes(raw_toml),
                },
                JSON_FILENAME: {
                    "bytes": len(normalized),
                    "sha256": sha256_bytes(normalized),
                },
                HEADER_FILENAME: {"bytes": len(header), "sha256": sha256_bytes(header)},
            },
        }
    )
    return {
        RAW_FILENAME: raw_toml,
        JSON_FILENAME: normalized,
        HEADER_FILENAME: header,
        MANIFEST_FILENAME: canonical_json_bytes(manifest),
    }


def _existing_provenance(output_dir: Path) -> dict[str, Any]:
    path = output_dir / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterError(
            f"cannot load {path}; use --refresh to create provenance"
        ) from exc
    _expect_keys(
        manifest,
        "manifest",
        (
            "schema_version",
            "method",
            "source",
            "inspection",
            "exporter",
            "cross_check",
            "generator",
            "outputs",
        ),
    )
    return {
        "source": manifest["source"],
        "inspection": manifest["inspection"],
        "exporter": manifest["exporter"],
        "cross_check": manifest["cross_check"],
    }


def write_or_check(
    output_dir: Path, artifacts: Mapping[str, bytes], *, check: bool
) -> None:
    """Write generated artifacts or reject stale files in check mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = []
    for filename, content in artifacts.items():
        path = output_dir / filename
        if check:
            try:
                current = path.read_bytes()
            except OSError:
                current = None
            if current != content:
                stale.append(filename)
        else:
            path.write_bytes(content)
    if stale:
        raise ParameterError(
            "generated GFN1 parameter files are stale: "
            + ", ".join(stale)
            + "; regenerate with tools/parameters/generate_gfn1.py"
        )


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/parameters"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tblite", type=Path)
    parser.add_argument("--tblite-source", type=Path)
    parser.add_argument("--tblite-revision", default="HEAD")
    parser.add_argument("--tblite-inspection-revision")
    parser.add_argument("--dxtb-source", type=Path)
    parser.add_argument("--dxtb-revision", default="HEAD")
    parser.add_argument("--dxtb-path", default=DEFAULT_DXTB_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line parameter generation workflow."""
    arguments = _arguments(argv)
    try:
        locale.setlocale(locale.LC_ALL, "C")
        if arguments.refresh:
            if None in (
                arguments.tblite,
                arguments.tblite_source,
                arguments.dxtb_source,
            ):
                raise ParameterError(
                    "--refresh requires --tblite, --tblite-source, and --dxtb-source"
                )
            if arguments.check:
                raise ParameterError("--refresh and --check are mutually exclusive")
            raw_toml, exporter = _export_tblite(arguments.tblite)
            provenance = {
                "source": _source_provenance(
                    arguments.tblite_source.resolve(), arguments.tblite_revision
                ),
                "exporter": exporter,
                "cross_check": _dxtb_provenance(
                    arguments.dxtb_source.resolve(),
                    arguments.dxtb_revision,
                    arguments.dxtb_path,
                    raw_toml,
                ),
            }
            if arguments.tblite_inspection_revision is None:
                raise ParameterError("--refresh requires --tblite-inspection-revision")
            provenance["inspection"] = _inspection_provenance(
                arguments.tblite_source.resolve(),
                provenance["source"]["revision"],
                arguments.tblite_inspection_revision,
            )
        else:
            raw_toml = (arguments.output_dir / RAW_FILENAME).read_bytes()
            provenance = _existing_provenance(arguments.output_dir)
        artifacts = build_artifacts(raw_toml, provenance)
        write_or_check(arguments.output_dir, artifacts, check=arguments.check)
    except (OSError, ParameterError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
