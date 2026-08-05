#!/usr/bin/env python3
"""Generate deterministic GFN2-xTB parameter artifacts from tblite.

The tblite TOML export is the source of truth.  This module validates that
export before producing a normalized JSON document and a compact C++ header.
Keeping validation here prevents a newly added upstream field from being
silently omitted from gpuxtb's runtime tables.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import locale
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import tomllib

SCHEMA_VERSION = 1
METHOD = "gfn2-xtb"
UPSTREAM_REPOSITORY = "https://github.com/tblite/tblite"
UPSTREAM_LICENSE = "LGPL-3.0-or-later"
RAW_FILENAME = "gfn2.toml"
JSON_FILENAME = "gfn2.json"
HEADER_FILENAME = "gfn2.hpp"
MANIFEST_FILENAME = "manifest.json"

ELEMENT_SYMBOLS = (
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
ATOMIC_NUMBER = {symbol: index for index, symbol in enumerate(ELEMENT_SYMBOLS, 1)}
ANGULAR_MOMENTUM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
SHELL_RE = re.compile(r"^([1-9][0-9]*)([spdfg])$")

ELEMENT_SCALARS = (
    "gam",
    "gam3",
    "zeff",
    "arep",
    "xbond",
    "en",
    "dkernel",
    "qkernel",
    "mprad",
    "mpvcn",
)
ELEMENT_SHELL_ARRAYS = (
    "shells",
    "levels",
    "slater",
    "ngauss",
    "refocc",
    "shpoly",
    "kcn",
    "lgam",
)

# These files define serialization and GFN2 parameters.  The source digest is
# computed from committed blobs at the recorded revision, not from mtimes or a
# potentially dirty worktree.
SOURCE_PATHS = (
    "app/driver_param.f90",
    "src/tblite/param.f90",
    "src/tblite/param",
    "src/tblite/xtb/gfn2.f90",
)


class ParameterError(ValueError):
    """Raised when an upstream export cannot be represented losslessly."""


def sha256_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest for *content*."""

    return hashlib.sha256(content).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with stable UTF-8, key ordering, and line endings."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ParameterError(f"{location} must be a TOML table")
    return value


def _expect_keys(
    table: Mapping[str, Any],
    location: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    missing = sorted(required_set - table.keys())
    unknown = sorted(table.keys() - required_set - optional_set)
    if missing:
        raise ParameterError(f"{location} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ParameterError(
            f"{location} has unsupported fields: {', '.join(unknown)}; "
            "bump the gpuxtb parameter schema before accepting them"
        )


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ParameterError(f"{location} must be finite")
    return result


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{location} must be an integer")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ParameterError(f"{location} must be a boolean")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ParameterError(f"{location} must be a string")
    return value


def _shell_scale(shell: Mapping[str, Any], first: int, second: int) -> float:
    names = "spdfg"
    direct = names[first] + names[second]
    if direct in shell:
        return _number(shell[direct], f"hamiltonian.xtb.shell.{direct}")
    if first != second:
        # tblite's hamiltonian_record loader reads only canonical pairs with
        # increasing angular momentum (sp, sd, pd, ...) and defaults an
        # omitted cross term to the arithmetic mean of its two diagonals.
        # Its serializer omits exactly those default-valued cross terms, so
        # reconstructing the mean is necessary for a lossless param export.
        diagonal_first = _number(
            shell[names[first] * 2], f"hamiltonian.xtb.shell.{names[first] * 2}"
        )
        diagonal_second = _number(
            shell[names[second] * 2], f"hamiltonian.xtb.shell.{names[second] * 2}"
        )
        return 0.5 * (diagonal_first + diagonal_second)
    raise ParameterError(f"missing diagonal shell scale for angular momentum {first}")


def _normalize_hamiltonian(source: Mapping[str, Any]) -> dict[str, Any]:
    _expect_keys(source, "hamiltonian", ("xtb",))
    xtb = _mapping(source["xtb"], "hamiltonian.xtb")
    _expect_keys(
        xtb,
        "hamiltonian.xtb",
        ("wexp", "kpol", "enscale", "cn", "shell"),
        ("kpair",),
    )
    shell = _mapping(xtb["shell"], "hamiltonian.xtb.shell")
    # A GFN2 export has lmax=2.  tblite serializes only canonical pair names:
    # the lower angular momentum is first, and reverse aliases are not read.
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

    diagonal_momenta = [
        momentum for label, momentum in ANGULAR_MOMENTUM.items() if label * 2 in shell
    ]
    if diagonal_momenta != list(range(len(diagonal_momenta))):
        raise ParameterError(
            "shell scales must contain contiguous diagonals starting at s"
        )
    if diagonal_momenta != [0, 1, 2]:
        raise ParameterError("GFN2 requires exactly the s, p, and d shell diagonals")

    shell_pair_scale = []
    for first in diagonal_momenta:
        for second in range(first, len(diagonal_momenta)):
            shell_pair_scale.append(
                {
                    "angular_momenta": [first, second],
                    "value": _shell_scale(shell, first, second),
                }
            )

    pair_overrides = []
    pair_table = _mapping(xtb.get("kpair", {}), "hamiltonian.xtb.kpair")
    seen_pairs: set[tuple[int, int]] = set()
    for key, value in pair_table.items():
        symbols = key.split("-")
        if len(symbols) != 2 or any(symbol not in ATOMIC_NUMBER for symbol in symbols):
            raise ParameterError(f"invalid element pair key: {key}")
        atomic_numbers = tuple(sorted(ATOMIC_NUMBER[symbol] for symbol in symbols))
        if atomic_numbers in seen_pairs:
            raise ParameterError(f"duplicate symmetric element pair: {key}")
        seen_pairs.add(atomic_numbers)
        pair_overrides.append(
            {
                "atomic_numbers": list(atomic_numbers),
                "value": _number(value, f"hamiltonian.xtb.kpair.{key}"),
            }
        )
    pair_overrides.sort(key=lambda item: item["atomic_numbers"])

    coordination_number = _string(xtb["cn"], "hamiltonian.xtb.cn")
    if coordination_number != "dexp":
        raise ParameterError(
            f"unsupported GFN2 coordination-number model: {coordination_number}"
        )

    return {
        "wexp": _number(xtb["wexp"], "hamiltonian.xtb.wexp"),
        "kpol": _number(xtb["kpol"], "hamiltonian.xtb.kpol"),
        "enscale": _number(xtb["enscale"], "hamiltonian.xtb.enscale"),
        "coordination_number": coordination_number,
        "shell_pair_scale": shell_pair_scale,
        # tblite initializes every omitted element-pair scale to one.
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
        for field in ELEMENT_SHELL_ARRAYS[1:]:
            values = record[field]
            if not isinstance(values, list) or len(values) != shell_count:
                raise ParameterError(
                    f"{location}.{field} must contain exactly {shell_count} values"
                )

        shells = []
        for shell_index, label_value in enumerate(raw_shells):
            label = _string(label_value, f"{location}.shells[{shell_index}]")
            match = SHELL_RE.fullmatch(label)
            if match is None:
                raise ParameterError(f"unsupported shell label {label!r} in {location}")
            shells.append(
                {
                    "index": shell_index,
                    "principal_quantum_number": int(match.group(1)),
                    "angular_momentum": ANGULAR_MOMENTUM[match.group(2)],
                    "level": _number(
                        record["levels"][shell_index], f"{location}.levels"
                    ),
                    "slater": _number(
                        record["slater"][shell_index], f"{location}.slater"
                    ),
                    "ngauss": _integer(
                        record["ngauss"][shell_index], f"{location}.ngauss"
                    ),
                    "reference_occupation": _number(
                        record["refocc"][shell_index], f"{location}.refocc"
                    ),
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

        element = {
            "atomic_number": atomic_number,
            "symbol": symbol,
            "shells": shells,
        }
        element.update(
            {
                field: _number(record[field], f"{location}.{field}")
                for field in ELEMENT_SCALARS
            }
        )
        elements.append(element)
    return elements


def normalize_export(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a parsed tblite GFN2 TOML export."""

    source = _mapping(document, "root")
    _expect_keys(
        source,
        "root",
        (
            "meta",
            "hamiltonian",
            "dispersion",
            "repulsion",
            "charge",
            "thirdorder",
            "multipole",
            "element",
        ),
    )

    meta = _mapping(source["meta"], "meta")
    _expect_keys(meta, "meta", ("name", "reference", "version", "format"))

    dispersion_root = _mapping(source["dispersion"], "dispersion")
    _expect_keys(dispersion_root, "dispersion", ("d4",))
    dispersion = _mapping(dispersion_root["d4"], "dispersion.d4")
    _expect_keys(
        dispersion, "dispersion.d4", ("sc", "smooth", "s6", "s8", "a1", "a2", "s9")
    )

    repulsion_root = _mapping(source["repulsion"], "repulsion")
    _expect_keys(repulsion_root, "repulsion", ("effective",))
    repulsion = _mapping(repulsion_root["effective"], "repulsion.effective")
    _expect_keys(repulsion, "repulsion.effective", ("kexp", "klight"))

    charge_root = _mapping(source["charge"], "charge")
    _expect_keys(charge_root, "charge", ("effective",))
    charge = _mapping(charge_root["effective"], "charge.effective")
    _expect_keys(charge, "charge.effective", ("gexp", "average"))

    thirdorder_root = _mapping(source["thirdorder"], "thirdorder")
    _expect_keys(thirdorder_root, "thirdorder", ("shell",))
    thirdorder = _mapping(thirdorder_root["shell"], "thirdorder.shell")
    _expect_keys(thirdorder, "thirdorder.shell", ("s", "p", "d"))

    multipole_root = _mapping(source["multipole"], "multipole")
    _expect_keys(multipole_root, "multipole", ("damped",))
    multipole = _mapping(multipole_root["damped"], "multipole.damped")
    _expect_keys(
        multipole, "multipole.damped", ("dmp3", "dmp5", "kexp", "shift", "rmax")
    )

    charge_average = _string(charge["average"], "charge.effective.average")
    if charge_average != "arithmetic":
        raise ParameterError(f"unsupported GFN2 charge average: {charge_average}")

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
            "self_consistent": _boolean(dispersion["sc"], "dispersion.d4.sc"),
            "smooth": _boolean(dispersion["smooth"], "dispersion.d4.smooth"),
            **{
                key: _number(dispersion[key], f"dispersion.d4.{key}")
                for key in ("s6", "s8", "a1", "a2", "s9")
            },
        },
        "repulsion": {
            "kexp": _number(repulsion["kexp"], "repulsion.effective.kexp"),
            "klight": _number(repulsion["klight"], "repulsion.effective.klight"),
        },
        "charge": {
            "gexp": _number(charge["gexp"], "charge.effective.gexp"),
            "average": charge_average,
        },
        "thirdorder": {
            key: _number(thirdorder[key], f"thirdorder.shell.{key}")
            for key in ("s", "p", "d")
        },
        "multipole": {
            key: _number(multipole[key], f"multipole.damped.{key}")
            for key in ("dmp3", "dmp5", "kexp", "shift", "rmax")
        },
        "elements": _normalize_elements(_mapping(source["element"], "element")),
    }


def parse_and_normalize(raw_toml: bytes) -> dict[str, Any]:
    """Parse UTF-8 TOML and return the validated canonical representation."""

    try:
        text = raw_toml.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParameterError("tblite export is not UTF-8") from exc
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ParameterError(f"invalid tblite TOML export: {exc}") from exc
    return normalize_export(document)


def _cpp_float(value: float) -> str:
    # 17 significant digits round-trip every IEEE-754 binary64 value.  Python's
    # formatting is locale-independent, unlike iostream formatting by default.
    result = format(value, ".17g")
    if "." not in result and "e" not in result:
        result += ".0"
    return result


def _cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render_header(parameters: Mapping[str, Any], source_revision: str) -> bytes:
    """Render the canonical parameters as trivially copyable C++ tables."""

    elements = parameters["elements"]
    shells = [shell for element in elements for shell in element["shells"]]
    hamiltonian = parameters["hamiltonian"]
    shell_scale_entries = {
        tuple(entry["angular_momenta"]): entry["value"]
        for entry in hamiltonian["shell_pair_scale"]
    }
    shell_matrix = []
    shell_lmax = max(pair[1] for pair in shell_scale_entries)
    for first in range(shell_lmax + 1):
        for second in range(shell_lmax + 1):
            shell_matrix.append(shell_scale_entries[tuple(sorted((first, second)))])

    lines = [
        "// Generated by tools/parameters/generate_gfn2.py; do not edit.",
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
        "namespace gpuxtb::parameters::gfn2 {",
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
        "  std::array<double, 3> thirdorder_shell_scale;",
        "  double multipole_dmp3;",
        "  double multipole_dmp5;",
        "  double multipole_kexp;",
        "  double multipole_shift;",
        "  double multipole_rmax;",
        "  std::uint8_t coordination_number_model;  // 0 = dexp",
        "  std::uint8_t charge_average;  // 0 = arithmetic",
        "  bool dispersion_self_consistent;",
        "  bool dispersion_smooth;",
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
        "  std::uint8_t reserved;",
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
        "// All structures remain plain data so a CUDA/ROCm backend can copy the",
        "// exact host tables once per device without a backend-specific decoder.",
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
        f"  {_cpp_float(parameters['dispersion']['s6'])},",
        f"  {_cpp_float(parameters['dispersion']['s8'])},",
        f"  {_cpp_float(parameters['dispersion']['a1'])},",
        f"  {_cpp_float(parameters['dispersion']['a2'])},",
        f"  {_cpp_float(parameters['dispersion']['s9'])},",
        f"  {_cpp_float(parameters['repulsion']['kexp'])},",
        f"  {_cpp_float(parameters['repulsion']['klight'])},",
        f"  {_cpp_float(parameters['charge']['gexp'])},",
        "  {{"
        + ", ".join(
            _cpp_float(parameters["thirdorder"][key]) for key in ("s", "p", "d")
        )
        + "}},",
        f"  {_cpp_float(parameters['multipole']['dmp3'])},",
        f"  {_cpp_float(parameters['multipole']['dmp5'])},",
        f"  {_cpp_float(parameters['multipole']['kexp'])},",
        f"  {_cpp_float(parameters['multipole']['shift'])},",
        f"  {_cpp_float(parameters['multipole']['rmax'])},",
        "  0u,",
        "  0u,",
        f"  {str(parameters['dispersion']['self_consistent']).lower()},",
        f"  {str(parameters['dispersion']['smooth']).lower()},",
        "};",
        "",
        "inline constexpr std::array<ElementParameters, kElementCount> kElements{{",
    ]

    shell_offset = 0
    for element in elements:
        lines.append(
            "  {"
            + ", ".join(
                (
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
            )
            + "},"
        )
        shell_offset += len(element["shells"])
    lines.extend(
        (
            "}};",
            "",
            "inline constexpr std::array<ShellParameters, kShellCount> kShells{{",
        )
    )
    for shell in shells:
        lines.append(
            "  {"
            + ", ".join(
                (
                    f"{shell['principal_quantum_number']}u",
                    f"{shell['angular_momentum']}u",
                    f"{shell['ngauss']}u",
                    "0u",
                    _cpp_float(shell["level"]),
                    _cpp_float(shell["slater"]),
                    _cpp_float(shell["reference_occupation"]),
                    _cpp_float(shell["shell_polynomial"]),
                    _cpp_float(shell["coordination_number_scale"]),
                    _cpp_float(shell["shell_hubbard_scale"]),
                )
            )
            + "},"
        )
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
            "[[nodiscard]] constexpr const ElementParameters* find_element(",
            "    std::uint32_t atomic_number) noexcept {",
            "  return atomic_number >= 1u && atomic_number <= kElementCount",
            "             ? &kElements[atomic_number - 1u]",
            "             : nullptr;",
            "}",
            "",
            "[[nodiscard]] constexpr double pair_scale(std::uint32_t first,",
            "                                          std::uint32_t second) noexcept {",
            "  for (const auto& override_value : kPairScaleOverrides) {",
            "    if ((override_value.first_atomic_number == first &&",
            "         override_value.second_atomic_number == second) ||",
            "        (override_value.first_atomic_number == second &&",
            "         override_value.second_atomic_number == first)) {",
            "      return override_value.value;",
            "    }",
            "  }",
            "  return kGlobal.pair_scale_default;",
            "}",
            "",
            "}  // namespace gpuxtb::parameters::gfn2",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C", "LANG": "C"})
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "stdout", None) or ""
        raise ParameterError(f"command failed: {' '.join(command)}\n{output}") from exc
    return completed.stdout.strip()


def _git(source_dir: Path, *arguments: str) -> str:
    return _run(("git", "-C", str(source_dir), *arguments))


def _source_provenance(source_dir: Path, revision_spec: str) -> dict[str, Any]:
    revision = _git(source_dir, "rev-parse", f"{revision_spec}^{{commit}}")
    paths_text = _git(
        source_dir,
        "ls-tree",
        "-r",
        "--name-only",
        revision,
        "--",
        *SOURCE_PATHS,
    )
    paths = sorted(filter(None, paths_text.splitlines()))
    if not paths:
        raise ParameterError(f"no tblite parameter sources found in {source_dir}")

    digest = hashlib.sha256()
    for path in paths:
        content = subprocess.run(
            ("git", "-C", str(source_dir), "show", f"{revision}:{path}"),
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        encoded_path = path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "little"))
        digest.update(encoded_path)
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
    with tempfile.TemporaryDirectory(prefix="gpuxtb-gfn2-") as temporary_directory:
        output = Path(temporary_directory) / RAW_FILENAME
        _run(
            (
                str(executable),
                "param",
                "--method",
                "gfn2",
                "--output",
                str(output),
            )
        )
        raw_toml = output.read_bytes()
    # Normalize line endings only; numeric text remains exactly as tblite wrote it.
    raw_toml = raw_toml.replace(b"\r\n", b"\n")
    if not raw_toml.endswith(b"\n"):
        raw_toml += b"\n"
    return raw_toml, {
        "version": version,
        "executable_sha256": sha256_bytes(executable.read_bytes()),
    }


def _generator_sha256() -> str:
    return sha256_bytes(Path(__file__).resolve().read_bytes())


def _manifest(
    provenance: Mapping[str, Any],
    raw_toml: bytes,
    normalized_json: bytes,
    header: bytes,
) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(provenance))
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "generator": {
                "path": "tools/parameters/generate_gfn2.py",
                "sha256": _generator_sha256(),
            },
            "outputs": {
                RAW_FILENAME: {
                    "bytes": len(raw_toml),
                    "sha256": sha256_bytes(raw_toml),
                },
                JSON_FILENAME: {
                    "bytes": len(normalized_json),
                    "sha256": sha256_bytes(normalized_json),
                },
                HEADER_FILENAME: {
                    "bytes": len(header),
                    "sha256": sha256_bytes(header),
                },
            },
        }
    )
    return manifest


def build_artifacts(raw_toml: bytes, provenance: Mapping[str, Any]) -> dict[str, bytes]:
    """Build every committed artifact from raw TOML and provenance metadata."""

    parameters = parse_and_normalize(raw_toml)
    source = _mapping(provenance.get("source"), "manifest.source")
    source_revision = _string(source.get("revision"), "manifest.source.revision")
    normalized_json = canonical_json_bytes(parameters)
    header = render_header(parameters, source_revision)
    manifest = _manifest(provenance, raw_toml, normalized_json, header)
    return {
        RAW_FILENAME: raw_toml,
        JSON_FILENAME: normalized_json,
        HEADER_FILENAME: header,
        MANIFEST_FILENAME: canonical_json_bytes(manifest),
    }


def _existing_provenance(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterError(
            f"cannot load {manifest_path}; use --refresh to create provenance"
        ) from exc
    _expect_keys(
        manifest,
        "manifest",
        ("schema_version", "method", "source", "exporter", "generator", "outputs"),
    )
    return {"source": manifest["source"], "exporter": manifest["exporter"]}


def write_or_check(
    output_dir: Path, artifacts: Mapping[str, bytes], *, check: bool
) -> None:
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
            "generated GFN2 parameter files are stale: "
            + ", ".join(stale)
            + "; regenerate with tools/parameters/generate_gfn2.py"
        )


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parameters"),
        help="artifact directory (default: data/parameters)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when committed artifacts are stale",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="run tblite and refresh raw data plus provenance",
    )
    parser.add_argument(
        "--tblite",
        type=Path,
        help="tblite executable used with --refresh",
    )
    parser.add_argument(
        "--tblite-source",
        type=Path,
        help="tblite Git checkout used with --refresh",
    )
    parser.add_argument(
        "--tblite-revision",
        default="HEAD",
        help="tblite commit/tag corresponding to the exporter (default: HEAD)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        # Force a stable process locale even when this module is called directly.
        locale.setlocale(locale.LC_ALL, "C")
        if arguments.refresh:
            if arguments.tblite is None or arguments.tblite_source is None:
                raise ParameterError("--refresh requires --tblite and --tblite-source")
            if arguments.check:
                raise ParameterError("--refresh and --check are mutually exclusive")
            raw_toml, exporter = _export_tblite(arguments.tblite)
            provenance = {
                "source": _source_provenance(
                    arguments.tblite_source.resolve(), arguments.tblite_revision
                ),
                "exporter": exporter,
            }
        else:
            raw_toml = (arguments.output_dir / RAW_FILENAME).read_bytes()
            provenance = _existing_provenance(arguments.output_dir)

        artifacts = build_artifacts(raw_toml, provenance)
        write_or_check(arguments.output_dir, artifacts, check=arguments.check)
    except (OSError, ParameterError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
