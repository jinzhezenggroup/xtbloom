#!/usr/bin/env python3
"""Generate compact GFN1-D3 tables from pinned simple-dftd3 Git blobs.

The generated header contains only the elements supported by xTBloom GFN1
(atomic numbers 1--86).  Reading committed blobs instead of the source
worktree makes unrelated local files and edits irrelevant to the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import locale
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION = 1
ELEMENT_COUNT = 86
UPSTREAM_ELEMENT_COUNT = 103
MAX_REFERENCE_COUNT = 7
UPSTREAM_REVISION = "6f0b06fbfa8653a23ca55c453772ce3af4420706"
UPSTREAM_TREE = "4022a91c9a72260efa96aba379b7061dc295d7b1"
UPSTREAM_TAG = "v1.4.0"
UPSTREAM_REPOSITORY = "https://github.com/dftd3/simple-dftd3"
UPSTREAM_LICENSE = "LGPL-3.0-or-later"
MCTC_REPOSITORY = "https://github.com/grimme-lab/mctc-lib"
MCTC_REVISION = "aa89d4bf5c0076fbf169b59eeb9e30185db0e5a5"
MCTC_TREE = "c64c3f6936121d2445459b57f7904a94b276d0b5"
MCTC_TAG = "v0.5.1"
MCTC_LICENSE = "Apache-2.0"
HEADER_FILENAME = "gfn1_d3.hpp"
JSON_FILENAME = "gfn1_d3.json"
MANIFEST_FILENAME = "gfn1_d3_manifest.json"

SIMPLE_DFTD3_SOURCE_PATHS = (
    "src/dftd3/reference.f90",
    "src/dftd3/data/r4r2.f90",
    "src/dftd3/data/vdwrad.f90",
)
SIMPLE_DFTD3_LEGAL_PATHS = ("COPYING", "COPYING.LESSER")
MCTC_SOURCE_PATHS = (
    "src/mctc/io/constants.f90",
    "src/mctc/io/codata2018.f90",
    "src/mctc/io/convert.f90",
)


class D3DataError(ValueError):
    """Raised when pinned upstream D3 data cannot be represented exactly."""


def _git(source: Path, *arguments: str) -> str:
    """Run a read-only Git query against one source checkout."""
    return subprocess.check_output(
        ("git", "-C", str(source), *arguments), text=True, encoding="utf-8"
    ).strip()


def _git_blob(source: Path, revision: str, path: str) -> bytes:
    """Read one committed blob without consulting the source worktree."""
    return subprocess.check_output(
        ("git", "-C", str(source), "show", f"{revision}:{path}")
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_digest(sources: Mapping[str, bytes]) -> str:
    """Hash path/blob pairs so the manifest identifies every parsed input."""
    digest = hashlib.sha256()
    for path in sorted(sources):
        encoded_path = path.encode("utf-8")
        content = sources[path]
        digest.update(len(encoded_path).to_bytes(8, "little"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _without_comments(text: str) -> str:
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


def _numeric_tokens(text: str) -> list[float]:
    """Parse decimal Fortran literals after comments and kind tags are removed."""
    source = _without_comments(text).replace("_wp", "")
    return [
        float(token.replace("D", "E").replace("d", "e"))
        for token in re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?", source
        )
    ]


def _integer_tokens(text: str) -> list[int]:
    return [int(token) for token in re.findall(r"[-+]?\d+", _without_comments(text))]


def _array_body(source: str, declaration: str, *, reshape: bool = False) -> str:
    """Return the bracket initializer following one exact declaration pattern."""
    suffix = (
        r"\s*\[\s*&?(.*?)\]\s*,?\s*&?(?:\s*!.*)?\s*&?\s*\["
        if reshape
        else r"\s*\[\s*&?(.*?)\]"
    )
    match = re.search(
        declaration + suffix,
        source,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise D3DataError(
            "could not locate an expected upstream parameter array: " + declaration
        )
    return match.group(1)


def _parse_reference(source: str) -> tuple[list[int], list[float], list[float]]:
    counts = _integer_tokens(
        _array_body(
            source,
            r"number_of_references\s*\(\s*max_elem\s*\)\s*=\s*",
        )
    )
    if len(counts) != UPSTREAM_ELEMENT_COUNT:
        raise D3DataError(
            f"reference-count array has {len(counts)} entries, expected "
            f"{UPSTREAM_ELEMENT_COUNT}"
        )
    if any(count < 1 or count > MAX_REFERENCE_COUNT for count in counts):
        raise D3DataError("reference-count array contains an unsupported value")

    coordination = _numeric_tokens(
        _array_body(
            source,
            r"reference_cn\s*\(\s*max_ref\s*,\s*max_elem\s*\)\s*=\s*reshape\s*\(\s*",
            reshape=True,
        )
    )
    expected_cn = MAX_REFERENCE_COUNT * UPSTREAM_ELEMENT_COUNT
    if len(coordination) != expected_cn:
        raise D3DataError(
            f"reference-CN array has {len(coordination)} entries, expected "
            f"{expected_cn}"
        )

    assignments = list(
        re.finditer(
            r"c6ab_view\s*\(\s*(\d+)\s*:\s*(\d+)\s*\)\s*=\s*\[(.*?)\]",
            source,
            re.DOTALL | re.IGNORECASE,
        )
    )
    expected_c6 = (
        MAX_REFERENCE_COUNT
        * MAX_REFERENCE_COUNT
        * UPSTREAM_ELEMENT_COUNT
        * (UPSTREAM_ELEMENT_COUNT + 1)
        // 2
    )
    c6 = [0.0] * expected_c6
    next_start = 1
    for assignment in assignments:
        start = int(assignment.group(1))
        end = int(assignment.group(2))
        values = _numeric_tokens(assignment.group(3))
        if start != next_start or end < start or len(values) != end - start + 1:
            raise D3DataError("C6 assignments are not contiguous and length-exact")
        c6[start - 1 : end] = values
        next_start = end + 1
    if next_start != expected_c6 + 1:
        raise D3DataError(
            f"C6 assignments end at {next_start - 1}, expected {expected_c6}"
        )
    return counts, coordination, c6


def _parameter_array(source: str, name: str, expected: int) -> list[float]:
    values = _numeric_tokens(
        _array_body(
            source,
            rf"{name}\s*\(.*?\)\s*=\s*(?:aatoau\s*\*)?\s*",
        )
    )
    if len(values) != expected:
        raise D3DataError(f"{name} has {len(values)} entries, expected {expected}")
    return values


def _named_real(source: str, name: str) -> float:
    match = re.search(
        rf"\b{name}\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?)_wp",
        source,
        re.IGNORECASE,
    )
    if match is None:
        raise D3DataError(f"could not locate mctc-lib constant {name}")
    return float(match.group(1).replace("D", "E").replace("d", "e"))


def _angstrom_to_bohr(mctc_sources: Mapping[str, bytes]) -> float:
    """Evaluate mctc-lib v0.5.1's binary64 ``aatoau`` expression."""
    constants = mctc_sources["src/mctc/io/constants.f90"].decode("utf-8")
    codata = mctc_sources["src/mctc/io/codata2018.f90"].decode("utf-8")
    convert = mctc_sources["src/mctc/io/convert.f90"].decode("utf-8")
    required_fragments = (
        "hbar = codata%h/(2.0_wp*pi)",
        "bohr = hbar/(codata%me*codata%c*codata%alpha)",
        "autoaa = bohr * 1e10_wp",
        "aatoau = 1.0_wp/autoaa",
    )
    if any(fragment not in convert for fragment in required_fragments):
        raise D3DataError("mctc-lib length-conversion expression has changed")
    pi = _named_real(constants, "pi")
    planck = _named_real(codata, "Planck_constant")
    speed = _named_real(codata, "speed_of_light_in_vacuum")
    fine_structure = _named_real(codata, "fine_structure_constant")
    electron_mass = _named_real(codata, "electron_mass")
    hbar = planck / (2.0 * pi)
    bohr_meters = hbar / (electron_mass * speed * fine_structure)
    value = 1.0 / (bohr_meters * 1.0e10)
    if value != 1.8897261246204404:
        raise D3DataError(
            "mctc-lib aatoau no longer matches the reviewed binary64 value"
        )
    return value


def build_tables(
    d3_sources: Mapping[str, bytes], mctc_sources: Mapping[str, bytes]
) -> dict[str, Any]:
    """Extract compact valid-reference D3 tables for atomic numbers 1--86."""
    reference = d3_sources["src/dftd3/reference.f90"].decode("utf-8")
    counts, upstream_cn, upstream_c6 = _parse_reference(reference)
    r4_over_r2 = _parameter_array(
        d3_sources["src/dftd3/data/r4r2.f90"].decode("utf-8"),
        "r4_over_r2",
        118,
    )
    upstream_vdw = _parameter_array(
        d3_sources["src/dftd3/data/vdwrad.f90"].decode("utf-8"),
        "vdwrad",
        UPSTREAM_ELEMENT_COUNT * (UPSTREAM_ELEMENT_COUNT + 1) // 2,
    )
    angstrom_to_bohr = _angstrom_to_bohr(mctc_sources)

    elements: list[dict[str, int]] = []
    coordination_numbers: list[float] = []
    for atomic_number in range(1, ELEMENT_COUNT + 1):
        count = counts[atomic_number - 1]
        offset = len(coordination_numbers)
        begin = (atomic_number - 1) * MAX_REFERENCE_COUNT
        valid = upstream_cn[begin : begin + count]
        padding = upstream_cn[begin + count : begin + MAX_REFERENCE_COUNT]
        if any(value != -1.0 for value in padding):
            raise D3DataError(
                f"element {atomic_number} has non-sentinel reference-CN padding"
            )
        coordination_numbers.extend(valid)
        elements.append({"reference_offset": offset, "reference_count": count})

    pair_records: list[dict[str, int]] = []
    c6: list[float] = []
    vdw_radii: list[float] = []
    for second in range(1, ELEMENT_COUNT + 1):
        second_count = counts[second - 1]
        for first in range(1, second + 1):
            first_count = counts[first - 1]
            pair_index = first + second * (second - 1) // 2
            c6_offset = len(c6)
            for second_reference in range(second_count):
                for first_reference in range(first_count):
                    upstream_index = (
                        second_reference
                        + MAX_REFERENCE_COUNT * first_reference
                        + MAX_REFERENCE_COUNT * MAX_REFERENCE_COUNT * (pair_index - 1)
                    )
                    c6.append(upstream_c6[upstream_index])
            pair_records.append(
                {
                    "c6_offset": c6_offset,
                    "first_reference_count": first_count,
                    "second_reference_count": second_count,
                }
            )
            vdw_radii.append(upstream_vdw[pair_index - 1] * angstrom_to_bohr)

    if any(value < 0.0 or not math.isfinite(value) for value in c6):
        raise D3DataError("valid-reference C6 table contains a negative value")
    r4r2 = [
        math.sqrt(0.5 * r4_over_r2[z - 1] * math.sqrt(float(z)))
        for z in range(1, ELEMENT_COUNT + 1)
    ]
    return {
        "angstrom_to_bohr": angstrom_to_bohr,
        "elements": elements,
        "coordination_numbers": coordination_numbers,
        "pair_records": pair_records,
        "c6": c6,
        "r4r2": r4r2,
        "vdw_radii": vdw_radii,
    }


def validate_tables(value: object) -> dict[str, Any]:
    """Validate the normalized JSON schema and every packed-table invariant."""
    if not isinstance(value, dict):
        raise D3DataError("normalized GFN1-D3 data root must be an object")
    expected_keys = {
        "angstrom_to_bohr",
        "elements",
        "coordination_numbers",
        "pair_records",
        "c6",
        "r4r2",
        "vdw_radii",
    }
    if set(value) != expected_keys:
        raise D3DataError("normalized GFN1-D3 data has unexpected fields")

    def finite_numbers(name: str, expected: int, *, positive: bool) -> list[float]:
        items = value[name]
        if not isinstance(items, list) or len(items) != expected:
            raise D3DataError(f"{name} must contain exactly {expected} values")
        result = []
        for item in items:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise D3DataError(f"{name} contains a non-numeric value")
            number = float(item)
            if not math.isfinite(number) or (positive and number <= 0.0):
                raise D3DataError(f"{name} contains an invalid value")
            result.append(number)
        return result

    conversion = value["angstrom_to_bohr"]
    if isinstance(conversion, bool) or not isinstance(conversion, (int, float)):
        raise D3DataError("angstrom_to_bohr must be numeric")
    if float(conversion) != 1.8897261246204404:
        raise D3DataError("angstrom_to_bohr differs from the pinned conversion")

    elements = value["elements"]
    if not isinstance(elements, list) or len(elements) != ELEMENT_COUNT:
        raise D3DataError(f"elements must contain exactly {ELEMENT_COUNT} records")
    reference_offset = 0
    normalized_elements = []
    for atomic_number, item in enumerate(elements, 1):
        if not isinstance(item, dict) or set(item) != {
            "reference_offset",
            "reference_count",
        }:
            raise D3DataError(f"element {atomic_number} record has unexpected fields")
        offset = item["reference_offset"]
        count = item["reference_count"]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or offset != reference_offset
            or count < 1
            or count > MAX_REFERENCE_COUNT
        ):
            raise D3DataError(f"element {atomic_number} has invalid reference packing")
        normalized_elements.append(
            {"reference_offset": offset, "reference_count": count}
        )
        reference_offset += count
    coordination = finite_numbers(
        "coordination_numbers", reference_offset, positive=False
    )

    pair_count = ELEMENT_COUNT * (ELEMENT_COUNT + 1) // 2
    pairs = value["pair_records"]
    if not isinstance(pairs, list) or len(pairs) != pair_count:
        raise D3DataError(f"pair_records must contain exactly {pair_count} records")
    normalized_pairs = []
    c6_offset = 0
    pair_index = 0
    for second in range(1, ELEMENT_COUNT + 1):
        for first in range(1, second + 1):
            item = pairs[pair_index]
            expected = {
                "c6_offset": c6_offset,
                "first_reference_count": normalized_elements[first - 1][
                    "reference_count"
                ],
                "second_reference_count": normalized_elements[second - 1][
                    "reference_count"
                ],
            }
            if item != expected:
                raise D3DataError(
                    f"element pair ({first}, {second}) has invalid C6 packing"
                )
            normalized_pairs.append(expected)
            c6_offset += (
                expected["first_reference_count"] * expected["second_reference_count"]
            )
            pair_index += 1
    c6 = finite_numbers("c6", c6_offset, positive=True)
    r4r2 = finite_numbers("r4r2", ELEMENT_COUNT, positive=True)
    vdw = finite_numbers("vdw_radii", pair_count, positive=True)
    return {
        "angstrom_to_bohr": float(conversion),
        "elements": normalized_elements,
        "coordination_numbers": coordination,
        "pair_records": normalized_pairs,
        "c6": c6,
        "r4r2": r4r2,
        "vdw_radii": vdw,
    }


def _cpp_float(value: float) -> str:
    if not math.isfinite(value):
        raise D3DataError("generated D3 data contains NaN or infinity")
    text = format(value, ".17g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def _format_array(values: Sequence[float], *, columns: int = 4) -> str:
    rendered = [_cpp_float(value) for value in values]
    return "\n".join(
        "    " + ", ".join(rendered[start : start + columns]) + ","
        for start in range(0, len(rendered), columns)
    )


def render_header(
    tables: Mapping[str, Any], source_digest: str, mctc_digest: str
) -> bytes:
    """Render trivially-copyable D3 tables with constexpr symmetric accessors."""
    elements = tables["elements"]
    pairs = tables["pair_records"]
    element_rows = [
        f"    D3ElementData{{{item['reference_offset']}u, {item['reference_count']}u}},"
        for item in elements
    ]
    pair_rows = [
        "    D3PairData{"
        f"{item['c6_offset']}u, {item['first_reference_count']}u, "
        f"{item['second_reference_count']}u}},"
        for item in pairs
    ]
    header = f"""// Generated by tools/parameters/generate_gfn1_d3.py; do not edit.
// SPDX-License-Identifier: LGPL-3.0-or-later
// Numerical data derived from simple-dftd3 ({UPSTREAM_LICENSE}).
// simple-dftd3 revision: {UPSTREAM_REVISION}
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xtbloom::parameters::gfn1_d3 {{

inline constexpr std::uint32_t kSchemaVersion = {SCHEMA_VERSION}u;
inline constexpr char kSourceRevision[] = "{UPSTREAM_REVISION}";
inline constexpr char kSourceDigest[] = "{source_digest}";
inline constexpr char kMctcRevision[] = "{MCTC_REVISION}";
inline constexpr char kMctcSourceDigest[] = "{mctc_digest}";
inline constexpr std::size_t kElementCount = {len(elements)}u;
inline constexpr std::size_t kReferenceCount = {len(tables["coordination_numbers"])}u;
inline constexpr std::size_t kElementPairCount = {len(pairs)}u;
inline constexpr std::size_t kReferenceC6Count = {len(tables["c6"])}u;
inline constexpr double kAngstromToBohr = {_cpp_float(tables["angstrom_to_bohr"])};

struct D3ElementData {{
  std::uint16_t reference_offset;
  std::uint8_t reference_count;
}};

struct D3PairData {{
  std::uint32_t c6_offset;
  std::uint8_t first_reference_count;
  std::uint8_t second_reference_count;
}};

static_assert(std::is_trivially_copyable_v<D3ElementData>);
static_assert(std::is_trivially_copyable_v<D3PairData>);

inline constexpr std::array<D3ElementData, kElementCount> kElements{{{{
{chr(10).join(element_rows)}
}}}};

inline constexpr std::array<double, kReferenceCount> kReferenceCoordinationNumbers{{{{
{_format_array(tables["coordination_numbers"])}
}}}};

/* Pairs use first <= second and Fortran's packed triangular element order. */
inline constexpr std::array<D3PairData, kElementPairCount> kPairs{{{{
{chr(10).join(pair_rows)}
}}}};

/* Each pair block is row-major in (second reference, first reference). */
inline constexpr std::array<double, kReferenceC6Count> kReferenceC6{{{{
{_format_array(tables["c6"])}
}}}};

inline constexpr std::array<double, kElementCount> kR4R2{{{{
{_format_array(tables["r4r2"])}
}}}};

/* Packed symmetric pair radii in bohr, using the same order as kPairs. */
inline constexpr std::array<double, kElementPairCount> kVdwRadii{{{{
{_format_array(tables["vdw_radii"])}
}}}};

[[nodiscard]] constexpr const D3ElementData* find_element(
    std::uint32_t atomic_number) noexcept {{
  return atomic_number >= 1u && atomic_number <= kElementCount
             ? &kElements[atomic_number - 1u]
             : nullptr;
}}

[[nodiscard]] constexpr std::size_t pair_index(
    std::uint32_t first, std::uint32_t second) noexcept {{
  if (first > second) {{
    const auto temporary = first;
    first = second;
    second = temporary;
  }}
  return static_cast<std::size_t>(first - 1u) +
         static_cast<std::size_t>(second) * (second - 1u) / 2u;
}}

[[nodiscard]] constexpr const D3PairData* find_pair(
    std::uint32_t first, std::uint32_t second) noexcept {{
  return first >= 1u && first <= kElementCount && second >= 1u &&
                 second <= kElementCount
             ? &kPairs[pair_index(first, second)]
             : nullptr;
}}

[[nodiscard]] constexpr double reference_cn(
    std::uint32_t atomic_number, std::uint32_t reference) noexcept {{
  const auto* element = find_element(atomic_number);
  return element != nullptr && reference < element->reference_count
             ? kReferenceCoordinationNumbers[element->reference_offset + reference]
             : 0.0;
}}

[[nodiscard]] constexpr double reference_c6(
    std::uint32_t first, std::uint32_t first_reference,
    std::uint32_t second, std::uint32_t second_reference) noexcept {{
  if (first > second) {{
    const auto temporary_element = first;
    first = second;
    second = temporary_element;
    const auto temporary_reference = first_reference;
    first_reference = second_reference;
    second_reference = temporary_reference;
  }}
  const auto* pair = find_pair(first, second);
  if (pair == nullptr || first_reference >= pair->first_reference_count ||
      second_reference >= pair->second_reference_count) {{
    return 0.0;
  }}
  return kReferenceC6[pair->c6_offset +
                      second_reference * pair->first_reference_count +
                      first_reference];
}}

[[nodiscard]] constexpr double vdw_radius(
    std::uint32_t first, std::uint32_t second) noexcept {{
  return find_pair(first, second) != nullptr
             ? kVdwRadii[pair_index(first, second)]
             : 0.0;
}}

}}  // namespace xtbloom::parameters::gfn1_d3
"""
    return header.encode("utf-8")


def _source_records(
    source: Path, revision: str, paths: Sequence[str]
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    sources = {path: _git_blob(source, revision, path) for path in paths}
    records = [
        {
            "path": path,
            "git_blob": _git(source, "rev-parse", f"{revision}:{path}"),
            "bytes": len(sources[path]),
            "sha256": _sha256(sources[path]),
        }
        for path in paths
    ]
    return sources, records


def build_artifacts(
    d3_source: Path,
    d3_revision_spec: str,
    mctc_source: Path,
    mctc_revision_spec: str,
) -> dict[str, bytes]:
    """Build the header and manifest from exact committed upstream objects."""
    d3_revision = _git(d3_source, "rev-parse", f"{d3_revision_spec}^{{commit}}")
    mctc_revision = _git(mctc_source, "rev-parse", f"{mctc_revision_spec}^{{commit}}")
    if d3_revision != UPSTREAM_REVISION:
        raise D3DataError(
            f"simple-dftd3 revision {d3_revision} is not the reviewed "
            f"{UPSTREAM_REVISION}"
        )
    if mctc_revision != MCTC_REVISION:
        raise D3DataError(
            f"mctc-lib revision {mctc_revision} is not the reviewed {MCTC_REVISION}"
        )
    d3_tree = _git(d3_source, "rev-parse", f"{d3_revision}^{{tree}}")
    mctc_tree = _git(mctc_source, "rev-parse", f"{mctc_revision}^{{tree}}")
    if d3_tree != UPSTREAM_TREE or mctc_tree != MCTC_TREE:
        raise D3DataError("one reviewed upstream tree no longer matches its pin")

    d3_sources, d3_records = _source_records(
        d3_source, d3_revision, SIMPLE_DFTD3_SOURCE_PATHS
    )
    _legal_sources, legal_records = _source_records(
        d3_source, d3_revision, SIMPLE_DFTD3_LEGAL_PATHS
    )
    mctc_sources, mctc_records = _source_records(
        mctc_source, mctc_revision, MCTC_SOURCE_PATHS
    )
    tables = validate_tables(build_tables(d3_sources, mctc_sources))
    d3_digest = _source_digest(d3_sources)
    mctc_digest = _source_digest(mctc_sources)
    normalized = (
        json.dumps(tables, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    header = render_header(tables, d3_digest, mctc_digest)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "gfn1-d3",
        "generator": {
            "path": "tools/parameters/generate_gfn1_d3.py",
            "sha256": _sha256(Path(__file__).resolve().read_bytes()),
        },
        "source": {
            "repository": UPSTREAM_REPOSITORY,
            "tag": UPSTREAM_TAG,
            "revision": d3_revision,
            "tree": d3_tree,
            "license": UPSTREAM_LICENSE,
            "source_digest": d3_digest,
            "parsed_sources": d3_records,
            "legal_files": legal_records,
        },
        "unit_conversion": {
            "repository": MCTC_REPOSITORY,
            "tag": MCTC_TAG,
            "revision": mctc_revision,
            "tree": mctc_tree,
            "license": MCTC_LICENSE,
            "source_digest": mctc_digest,
            "sources": mctc_records,
            "angstrom_to_bohr": tables["angstrom_to_bohr"],
            "contract": "Evaluate mctc-lib aatoau from CODATA 2018 constants.",
        },
        "representation": {
            "element_count": len(tables["elements"]),
            "reference_count": len(tables["coordination_numbers"]),
            "element_pair_count": len(tables["pair_records"]),
            "reference_c6_count": len(tables["c6"]),
            "c6_pair_layout": " ".join(
                (
                    "packed first<=second; row-major",
                    "(second reference, first reference)",
                )
            ),
            "vdw_radius_unit": "bohr",
            "r4r2_expression": "sqrt(0.5 * upstream_r4_over_r2[Z] * sqrt(Z))",
        },
        "outputs": {
            JSON_FILENAME: {"bytes": len(normalized), "sha256": _sha256(normalized)},
            HEADER_FILENAME: {"bytes": len(header), "sha256": _sha256(header)},
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return {
        JSON_FILENAME: normalized,
        HEADER_FILENAME: header,
        MANIFEST_FILENAME: manifest_bytes,
    }


def build_offline_artifacts(output_dir: Path) -> dict[str, bytes]:
    """Re-render generated files from retained normalized data and provenance."""
    try:
        tables = validate_tables(
            json.loads((output_dir / JSON_FILENAME).read_text(encoding="utf-8"))
        )
        manifest = json.loads(
            (output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        source_digest = manifest["source"]["source_digest"]
        mctc_digest = manifest["unit_conversion"]["source_digest"]
    except (KeyError, json.JSONDecodeError, OSError, TypeError) as exc:
        raise D3DataError("cannot load retained GFN1-D3 data and provenance") from exc
    if not isinstance(source_digest, str) or not isinstance(mctc_digest, str):
        raise D3DataError("GFN1-D3 manifest contains invalid source digests")
    normalized = (
        json.dumps(tables, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    header = render_header(tables, source_digest, mctc_digest)
    refreshed = dict(manifest)
    refreshed["generator"] = {
        "path": "tools/parameters/generate_gfn1_d3.py",
        "sha256": _sha256(Path(__file__).resolve().read_bytes()),
    }
    refreshed["outputs"] = {
        JSON_FILENAME: {"bytes": len(normalized), "sha256": _sha256(normalized)},
        HEADER_FILENAME: {"bytes": len(header), "sha256": _sha256(header)},
    }
    manifest_bytes = (json.dumps(refreshed, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return {
        JSON_FILENAME: normalized,
        HEADER_FILENAME: header,
        MANIFEST_FILENAME: manifest_bytes,
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
        raise D3DataError(
            "generated GFN1-D3 files are stale: "
            + ", ".join(stale)
            + "; regenerate with tools/parameters/generate_gfn1_d3.py"
        )


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simple-dftd3-source", type=Path)
    parser.add_argument("--simple-dftd3-revision", default=UPSTREAM_REVISION)
    parser.add_argument("--mctc-source", type=Path)
    parser.add_argument("--mctc-revision", default=MCTC_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("data/parameters"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line GFN1-D3 generation workflow."""
    arguments = _arguments(argv)
    try:
        locale.setlocale(locale.LC_ALL, "C")
        if arguments.refresh:
            if arguments.check:
                raise D3DataError("--refresh and --check are mutually exclusive")
            if arguments.simple_dftd3_source is None or arguments.mctc_source is None:
                raise D3DataError(
                    "--refresh requires --simple-dftd3-source and --mctc-source"
                )
            artifacts = build_artifacts(
                arguments.simple_dftd3_source.resolve(),
                arguments.simple_dftd3_revision,
                arguments.mctc_source.resolve(),
                arguments.mctc_revision,
            )
        else:
            if (
                arguments.simple_dftd3_source is not None
                or arguments.mctc_source is not None
            ):
                raise D3DataError("source checkouts are accepted only with --refresh")
            artifacts = build_offline_artifacts(arguments.output_dir)
        write_or_check(arguments.output_dir, artifacts, check=arguments.check)
    except (D3DataError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
