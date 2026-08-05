#!/usr/bin/env python3
"""Generate packed GFN2-D4 reference data from a pinned dftd4 revision.

The runtime intentionally does not depend on Fortran or libdftd4.  This tool
extracts the LGPL-3.0-or-later reference-system data, constructs the GFN2
charge-model reference polarizabilities, and performs the Casimir--Polder
quadrature once at generation time.  The resulting packed reference C6 matrix
is directly usable by CPU, CUDA, and future ROCm kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

DFTD4_REPOSITORY = "https://github.com/dftd4/dftd4"
DFTD4_LICENSE = "LGPL-3.0-or-later"
MCTC_REVISION = "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d"
MCTC_TREE = "ff26e808ccd00be0ca221656c8de9ce0da4c9265"
ELEMENT_COUNT = 86
ANGSTROM_TO_BOHR = 1.8897261246204404
GA = 3.0
GC = 2.0
FREQUENCIES = (
    0.000001,
    0.050000,
    0.100000,
    0.200000,
    0.300000,
    0.400000,
    0.500000,
    0.600000,
    0.700000,
    0.800000,
    0.900000,
    1.000000,
    1.200000,
    1.400000,
    1.600000,
    1.800000,
    2.000000,
    2.500000,
    3.000000,
    4.000000,
    5.000000,
    7.500000,
    10.000000,
)
SOURCE_PATHS = (
    "src/dftd4/reference.inc",
    "src/dftd4/reference.f90",
    "src/dftd4/model/d4.f90",
    "src/dftd4/model/utils.f90",
    "src/dftd4/data/covrad.f90",
    "src/dftd4/data/en.f90",
    "src/dftd4/data/zeff.f90",
    "src/dftd4/data/hardness.f90",
    "src/dftd4/data/r4r2.f90",
)


class D4DataError(ValueError):
    """Raised when pinned upstream data cannot be represented losslessly."""


def git_show(git_dir: Path, revision: str, path: str) -> str:
    """Read one committed UTF-8 blob without consulting a working tree."""

    return subprocess.check_output(
        ["git", f"--git-dir={git_dir}", "show", f"{revision}:{path}"],
        text=True,
        encoding="utf-8",
    )


def git_value(git_dir: Path, *arguments: str) -> str:
    """Run a read-only Git query and return its stripped scalar output."""

    return subprocess.check_output(
        ["git", f"--git-dir={git_dir}", *arguments], text=True, encoding="utf-8"
    ).strip()


def source_digest(sources: dict[str, str]) -> str:
    """Hash path/blob pairs so the manifest identifies every parsed input."""

    digest = hashlib.sha256()
    for path in sorted(sources):
        encoded_path = path.encode("utf-8")
        encoded_blob = sources[path].encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "little"))
        digest.update(encoded_path)
        digest.update(len(encoded_blob).to_bytes(8, "little"))
        digest.update(encoded_blob)
    return digest.hexdigest()


def numeric_tokens(text: str) -> list[float]:
    """Parse decimal Fortran literals after comments and kind tags are removed."""

    uncommented = "\n".join(line.split("!", 1)[0] for line in text.splitlines())
    uncommented = uncommented.replace("_wp", "").replace("D", "E").replace("d", "e")
    return [
        float(token)
        for token in re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", uncommented
        )
    ]


def parameter_array(source: str, name: str, expected: int = 118) -> list[float]:
    """Extract one ``name(max_elem) = [...]`` parameter array."""

    match = re.search(
        rf"\b{name}\s*\(\s*max_elem\s*\)\s*=\s*(?:aatoau\s*\*)?\s*\[(.*?)\]",
        source,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise D4DataError(f"could not locate {name} parameter array")
    values = numeric_tokens(match.group(1))
    if len(values) != expected:
        raise D4DataError(f"{name} has {len(values)} entries, expected {expected}")
    return values


def parse_reference_data(
    source: str,
) -> tuple[dict[tuple[str, int, int], float], dict[tuple[str, int, int], list[float]]]:
    """Parse scalar and rank-one ``data`` initializers from reference.inc."""

    scalars: dict[tuple[str, int, int], float] = {}
    scalar_pattern = re.compile(
        r"data\s+(\w+)\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)\s*/\s*"
        r"([-+0-9.Ee]+)(?:_wp)?\s*/",
        re.IGNORECASE,
    )
    for match in scalar_pattern.finditer(source):
        key = (match.group(1).lower(), int(match.group(2)), int(match.group(3) or 0))
        scalars[key] = float(match.group(4))

    arrays: dict[tuple[str, int, int], list[float]] = {}
    array_pattern = re.compile(
        r"data\s+(\w+)\s*\(\s*:\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)\s*/(.*?)/",
        re.IGNORECASE | re.DOTALL,
    )
    for match in array_pattern.finditer(source):
        key = (match.group(1).lower(), int(match.group(2)), int(match.group(3) or 0))
        arrays[key] = numeric_tokens(match.group(4))
    return scalars, arrays


def zeta(a: float, c: float, qref: float, qmod: float) -> float:
    """DFT-D4 charge scaling function used for GFN2 reference data."""

    if qmod < 0.0:
        return math.exp(a)
    return math.exp(a * (1.0 - math.exp(c * (1.0 - qref / qmod))))


def trapzd(values: list[float]) -> float:
    """Apply dftd4's fixed 23-point trapezoidal Casimir--Polder quadrature."""

    if len(values) != len(FREQUENCIES):
        raise D4DataError("dynamic-polarizability vector does not have 23 entries")
    total = 0.0
    for index, value in enumerate(values):
        if index == 0:
            weight = 0.5 * (FREQUENCIES[1] - FREQUENCIES[0])
        elif index + 1 == len(FREQUENCIES):
            weight = 0.5 * (FREQUENCIES[-1] - FREQUENCIES[-2])
        else:
            weight = 0.5 * (FREQUENCIES[index + 1] - FREQUENCIES[index - 1])
        total += weight * value
    return total


def format_double(value: float) -> str:
    """Emit a locale-independent round-trippable C++ binary64 literal."""

    if not math.isfinite(value):
        raise D4DataError("generated D4 data contains NaN or infinity")
    text = format(value, ".17g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def format_array(
    values: list[float] | list[int], indent: str = "    ", columns: int = 4
) -> str:
    """Format a constexpr initializer with bounded line length."""

    rendered = [
        format_double(value) if isinstance(value, float) else str(value)
        for value in values
    ]
    lines = []
    for start in range(0, len(rendered), columns):
        lines.append(indent + ", ".join(rendered[start : start + columns]) + ",")
    return "\n".join(lines)


def build_tables(
    sources: dict[str, str],
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], list[float]]:
    """Construct packed element/reference records and their dense C6 matrix."""

    scalars, arrays = parse_reference_data(sources["src/dftd4/reference.inc"])
    covalent = parameter_array(
        sources["src/dftd4/data/covrad.f90"], "covalent_rad_2009"
    )
    electronegativity = parameter_array(sources["src/dftd4/data/en.f90"], "pauling_en")
    effective_charge = parameter_array(
        sources["src/dftd4/data/zeff.f90"], "effective_nuclear_charge"
    )
    hardness = parameter_array(
        sources["src/dftd4/data/hardness.f90"], "chemical_hardness"
    )
    r4_over_r2 = parameter_array(sources["src/dftd4/data/r4r2.f90"], "r4_over_r2")

    elements: list[dict[str, float | int]] = []
    references: list[dict[str, float | int]] = []
    polarizabilities: list[list[float]] = []
    for atomic_number in range(1, ELEMENT_COUNT + 1):
        ref_count = int(scalars.get(("refn", atomic_number, 0), 0.0))
        if not 1 <= ref_count <= 7:
            raise D4DataError(
                f"element {atomic_number} has invalid reference count {ref_count}"
            )
        reference_offset = len(references)
        for reference_index in range(1, ref_count + 1):
            required_scalars = {
                name: scalars[(name, reference_index, atomic_number)]
                for name in ("refq", "refh", "hcount", "ascale", "refcovcn", "refsys")
            }
            raw_alpha = arrays[("alphaiw", reference_index, atomic_number)]
            if len(raw_alpha) != len(FREQUENCIES):
                raise D4DataError(
                    f"element {atomic_number} reference {reference_index} has invalid alpha data"
                )
            secondary = int(required_scalars["refsys"])
            if secondary <= 0:
                raise D4DataError("GFN2 D4 reference uses an invalid secondary system")
            secondary_alpha = arrays[("secaiw", secondary, 0)]
            secondary_scale = scalars[("sscale", secondary, 0)]
            secondary_z = effective_charge[secondary - 1]
            hydrogen_alpha_scale = zeta(
                GA,
                hardness[secondary - 1] * GC,
                secondary_z,
                required_scalars["refh"] + secondary_z,
            )
            alpha = [
                max(
                    required_scalars["ascale"]
                    * (
                        raw_alpha[index]
                        - required_scalars["hcount"]
                        * secondary_scale
                        * secondary_alpha[index]
                        * hydrogen_alpha_scale
                    ),
                    0.0,
                )
                for index in range(len(FREQUENCIES))
            ]
            rounded_cn = min(
                math.floor(scalars[("refcn", reference_index, atomic_number)] + 0.5), 19
            )
            duplicate_count = (1 if rounded_cn == 0 else 0) + sum(
                math.floor(scalars[("refcn", other, atomic_number)] + 0.5) == rounded_cn
                for other in range(1, ref_count + 1)
            )
            gaussian_count = duplicate_count * (duplicate_count + 1) // 2
            references.append(
                {
                    "coordination_number": required_scalars["refcovcn"],
                    "charge": required_scalars["refq"],
                    "gaussian_count": gaussian_count,
                }
            )
            polarizabilities.append(alpha)

        z = float(atomic_number)
        elements.append(
            {
                "reference_offset": reference_offset,
                "reference_count": ref_count,
                "covalent_radius": (4.0 / 3.0)
                * ANGSTROM_TO_BOHR
                * covalent[atomic_number - 1],
                "electronegativity": electronegativity[atomic_number - 1],
                "effective_charge": effective_charge[atomic_number - 1],
                "hardness": hardness[atomic_number - 1],
                "r4r2": math.sqrt(0.5 * r4_over_r2[atomic_number - 1] * math.sqrt(z)),
            }
        )

    c6: list[float] = []
    factor = 3.0 / math.pi
    for first in polarizabilities:
        for second in polarizabilities:
            c6.append(
                factor * trapzd([a * b for a, b in zip(first, second, strict=True)])
            )
    return elements, references, c6


def render_header(
    revision: str,
    digest: str,
    elements: list[dict[str, float | int]],
    references: list[dict[str, float | int]],
    c6: list[float],
) -> str:
    """Render the packed data as one header shared by host and device code."""

    element_rows = []
    for element in elements:
        element_rows.append(
            "    D4ElementData{"
            + ", ".join(
                (
                    str(element["reference_offset"]),
                    str(element["reference_count"]),
                    format_double(float(element["covalent_radius"])),
                    format_double(float(element["electronegativity"])),
                    format_double(float(element["effective_charge"])),
                    format_double(float(element["hardness"])),
                    format_double(float(element["r4r2"])),
                )
            )
            + "},"
        )
    reference_rows = []
    for reference in references:
        reference_rows.append(
            "    D4ReferenceData{"
            + ", ".join(
                (
                    format_double(float(reference["coordination_number"])),
                    format_double(float(reference["charge"])),
                    str(reference["gaussian_count"]),
                )
            )
            + "},"
        )
    return f"""// Generated by tools/parameters/generate_d4.py; do not edit.
// SPDX-License-Identifier: LGPL-3.0-or-later
// Numerical data derived from dftd4 ({DFTD4_LICENSE}).
// dftd4 revision: {revision}
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace gpuxtb::parameters::d4 {{

inline constexpr char kSourceRevision[] = "{revision}";
inline constexpr char kSourceDigest[] = "{digest}";
inline constexpr std::size_t kElementCount = {len(elements)}u;
inline constexpr std::size_t kReferenceCount = {len(references)}u;

struct D4ElementData {{
  std::uint16_t reference_offset;
  std::uint8_t reference_count;
  double covalent_radius;
  double electronegativity;
  double effective_charge;
  double hardness;
  double r4r2;
}};

struct D4ReferenceData {{
  double coordination_number;
  double charge;
  std::uint8_t gaussian_count;
}};

inline constexpr std::array<D4ElementData, kElementCount> kElements{{{{
{chr(10).join(element_rows)}
}}}};

inline constexpr std::array<D4ReferenceData, kReferenceCount> kReferences{{{{
{chr(10).join(reference_rows)}
}}}};

/* Row-major matrix indexed by packed global reference indices. */
inline constexpr std::array<double, kReferenceCount * kReferenceCount> kReferenceC6{{{{
{format_array(c6)}
}}}};

}}  // namespace gpuxtb::parameters::d4
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        path: git_show(args.source_git_dir, args.revision, path)
        for path in SOURCE_PATHS
    }
    digest = source_digest(sources)
    elements, references, c6 = build_tables(sources)
    header = render_header(args.revision, digest, elements, references, c6)
    source_records = [
        {
            "git_blob": git_value(
                args.source_git_dir, "rev-parse", f"{args.revision}:{path}"
            ),
            "path": path,
            "sha256": hashlib.sha256(sources[path].encode("utf-8")).hexdigest(),
            "size": len(sources[path].encode("utf-8")),
        }
        for path in SOURCE_PATHS
    ]
    manifest = {
        "generator": "tools/parameters/generate_d4.py",
        "license": DFTD4_LICENSE,
        "mctc_revision": MCTC_REVISION,
        "mctc_tree": MCTC_TREE,
        "reference_count": len(references),
        "repository": DFTD4_REPOSITORY,
        "revision": args.revision,
        "source_digest": digest,
        "sources": source_records,
        "tree": git_value(
            args.source_git_dir, "rev-parse", f"{args.revision}^{{tree}}"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "d4.hpp").write_text(header, encoding="utf-8", newline="\n")
    (args.output_dir / "d4_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
