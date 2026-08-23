#!/usr/bin/env python3
"""Generate and verify pinned tblite periodic GFN2 term fixtures.

The live path invokes only the standalone Fortran probe linked to the exact
pinned tblite build.  The default ``check`` path is fully offline: it verifies
all tracked hashes, array contracts, independent recompositions, finite-
difference evidence, and invariants without loading tblite or xTBloom.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA = "xtbloom-periodic-gfn2-term-fixture-v1"
PROBE_SCHEMA = "xtbloom-tblite-periodic-gfn2-term-probe-v1"
MODES = ("cn", "repulsion", "d4", "integrals-h0", "charge-ewald", "multipoles")
INPUT_ID = "water_skew"
COORDINATE_ATOM = 1
COORDINATE_AXIS = 0
STRAIN_ROW = 0
STRAIN_COLUMN = 0
FD_STEPS = (1.0e-4, 5.0e-5)
BUILD_ATTESTATION = {
    "build_id": "tblite-133f91ef-gfortran14-netlib-613ec662",
    "path": "data/conformance/periodic/tblite-build.json",
    "sha256": "f9a6a86a1ab6a11299e4dd2a30f2fd4ca2f714be5c76d2ec2b1ddd037641b782",
}
InputDocument = dict[str, Any]
Variant = Callable[[InputDocument, float], InputDocument]

SOURCE_RECORDS = {
    "tblite": {
        "revision": "133f91efb94b47f05848e1f86832f40a1accc385",
        "repository": "https://github.com/tblite/tblite",
        "files": {
            "src/tblite/coulomb/charge/effective.f90": {
                "git_blob": "67b91d63e13803e5bc297597e7312404bb1e576c",
                "sha256": (
                    "1d66415c1046e3b83bde1434f6e1ea5697a95362332c70a8db08b2668a07214a"
                ),
            },
            "src/tblite/coulomb/ewald.f90": {
                "git_blob": "363a3b824d57185d19df79a2d7ce09e382f54a65",
                "sha256": (
                    "561e46c89f7dd2551936faf13d1749de1712aa186ef4ca1e83a05f5168236b44"
                ),
            },
            "src/tblite/coulomb/multipole.f90": {
                "git_blob": "9c4f86dc242f7b161c8d9cddeff0df6c2bb72f75",
                "sha256": (
                    "16f464f878876e9948c16042af7665518dc2c86d98332b51f661f22dffe2a8b6"
                ),
            },
            "src/tblite/disp/d4.f90": {
                "git_blob": "55b110037de26e260b3c761c352248268081b90f",
                "sha256": (
                    "3eb1577c7504da68da0bb7f8345ea9945996f45939dcb26744c9d4d034ede0d8b"
                ),
            },
            "src/tblite/integral/native/integrals.f90": {
                "git_blob": "37a0c95d13c39aaf42c28c75bd264ae5efbaf7a3",
                "sha256": (
                    "39a44a9a03e5a5c8e0a70d5217e36ca50544e7e652e5dbeedb8fe95544341798"
                ),
            },
            "src/tblite/repulsion/effective.f90": {
                "git_blob": "236d646e365f229e9ab36976d00523290091e47a",
                "sha256": (
                    "41f3ff370a59daba5687d00c47b804c14c8af0837da4c51d3a83d594fe079ed5"
                ),
            },
            "src/tblite/xtb/gfn2.f90": {
                "git_blob": "1da6f7389bac5a8ade24276652c6da6f64b5978d",
                "sha256": (
                    "cfe0d99d744be38f8e25a3271734ac16ae06156999b6182059b63782e8e6b432"
                ),
            },
            "src/tblite/xtb/h0.f90": {
                "git_blob": "14edb1d731f7ba716a83db3bb7583bc245075e0b",
                "sha256": (
                    "40935bb3a80286fac01670e2b7945770fc4f2d1ed922e22e8b859ce4b283608a"
                ),
            },
        },
    },
    "mctc-lib": {
        "revision": "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d",
        "repository": "https://github.com/grimme-lab/mctc-lib",
        "files": {
            "src/mctc/ncoord/erf.f90": {
                "git_blob": "5ea3c2c4056c3eca51e76629e0e780061923f253",
                "sha256": (
                    "6bc976e03e290bc01bccfeee3016513e6154ef6267f152d36b7c99a7d706fea9"
                ),
            },
            "src/mctc/ncoord/erf/dftd4.f90": {
                "git_blob": "830098d3040230fafc730a59d4c611b8875e0462",
                "sha256": (
                    "3bc326ecfbd5a8b43cac9f021a1472df335f7ef041e2f33916906fa71c60a3ab"
                ),
            },
        },
    },
    "dftd4": {
        "revision": "6e1f59c3f39d919a2dbef0601d2576727c8b30e8",
        "repository": "https://github.com/dftd4/dftd4",
        "files": {
            "src/dftd4/model/d4.f90": {
                "git_blob": "33ed0ea918a81bc88a22110cc856d7988b18f956",
                "sha256": (
                    "dc421a2ccf76227ae028f45ad84efd1683505b803f926dfda24be1b5d223b220"
                ),
            },
            "src/dftd4/model/type.f90": {
                "git_blob": "f0b4a8465ac55e00f88a0fdbd23984669f7af58f",
                "sha256": (
                    "5c7c9170328386c2799036a0ee047525a2911367aba818ba161fb17b720338b3"
                ),
            },
            "src/dftd4/damping/rational.f90": {
                "git_blob": "9d359c95d2b67365b9e61ec33ad663b564e53334",
                "sha256": (
                    "449c052716272c1bb35ad0f41bb23b9a0df019de45c43eb308267c568724d396"
                ),
            },
            "src/dftd4/damping/atm.f90": {
                "git_blob": "c6162c97a380362a0bf4059bad7c24d7434f8d4d",
                "sha256": (
                    "7f7e718bfa0892bd48591258b00430e3f7327693275f757fa6e3b27947b4f252"
                ),
            },
            "src/dftd4/cutoff.f90": {
                "git_blob": "86a856b5e738f6c43bde5b0597f05987c9502de5",
                "sha256": (
                    "e9e4b2b6d0cfdb138ff304cf109e59a9549998d2b9008821b4c53a5361c01e1a"
                ),
            },
            "src/dftd4/disp.f90": {
                "git_blob": "73253cee0288803eedcea29f9704b6b9b2bbf4ce",
                "sha256": (
                    "e83b1b3cdf1bb2caa56d4ec5b7014b02eb41f362a160dba7e3adbf82dabe2cad"
                ),
            },
        },
    },
    "multicharge": {
        "revision": "6a5d63f9e9e29dcf13cc47cc27f33bf9015681bf",
        "repository": "https://github.com/grimme-lab/multicharge",
        "scientific_role": (
            "linked runtime provenance only; GFN2 qmod does not invoke EEQ"
        ),
    },
}


class FixtureError(RuntimeError):
    """Raised when fixture generation or verification fails."""


def repository_root() -> Path:
    """Return the root of the current checkout or linked worktree."""
    return Path(__file__).resolve().parents[3]


def corpus_root() -> Path:
    """Return the committed periodic term-corpus directory."""
    return repository_root() / "data/conformance/periodic/terms"


def oracle_root() -> Path:
    """Resolve the shared oracle cache without embedding a checkout path."""
    override = os.environ.get("XTBLOOM_TBLITE_ORACLE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root()),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip()).resolve().parent / "build/oracles"


def oracle_build_paths() -> tuple[Path, Path]:
    """Return the pinned dependency environment and tblite build roots."""
    root = oracle_root()
    environment_root = Path(
        os.environ.get("XTBLOOM_TBLITE_ORACLE_ENV", root / "tblite-133-env")
    ).resolve()
    build_root = Path(
        os.environ.get("XTBLOOM_TBLITE_ORACLE_BUILD", root / "tblite-133-build-gf143")
    ).resolve()
    return environment_root, build_root


def probe_environment() -> dict[str, str]:
    """Construct the deterministic runtime environment for the live probe."""
    environment_root, build_root = oracle_build_paths()
    env = os.environ.copy()
    loader_dirs = [str(build_root), str(environment_root / "lib")]
    inherited_loader_path = env.get("LD_LIBRARY_PATH")
    if inherited_loader_path:
        loader_dirs.append(inherited_loader_path)
    env.update(
        {
            "LD_LIBRARY_PATH": ":".join(loader_dirs),
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return env


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    return sha256_bytes(path.read_bytes())


def canonical_json(document: object) -> bytes:
    """Serialize a JSON-compatible value in the corpus canonical form."""
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def compact_json_sha256(document: object) -> str:
    """Hash a value using path-independent compact canonical JSON."""
    encoded = json.dumps(
        document, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return sha256_bytes(encoded)


def product(values: list[int]) -> int:
    """Return the product of integer array extents."""
    result = 1
    for value in values:
        result *= value
    return result


def parse_probe_output(raw: bytes) -> dict[str, Any]:
    """Parse the probe's lossless line protocol into canonical array records."""
    lines = raw.decode("ascii").splitlines()
    if len(lines) < 2 or lines[0] != f"SCHEMA {PROBE_SCHEMA}":
        raise FixtureError("probe output has the wrong schema")
    mode_tokens = lines[1].split()
    if len(mode_tokens) != 2 or mode_tokens[0] != "MODE":
        raise FixtureError("probe output lacks a mode header")
    mode = mode_tokens[1]
    arrays: dict[str, Any] = {}
    line = 2
    while line < len(lines):
        header = lines[line].split()
        line += 1
        if len(header) < 3 or header[0] not in {"REAL", "INTEGER"}:
            raise FixtureError(f"malformed probe array header at line {line}")
        kind, name = header[:2]
        rank = int(header[2])
        shape = [int(item) for item in header[3:]]
        if len(shape) != rank:
            raise FixtureError(f"rank/shape mismatch for {name}")
        count = product(shape) if shape else 1
        if line + count > len(lines):
            raise FixtureError(f"truncated values for {name}")
        if kind == "REAL":
            values = [float(item) for item in lines[line : line + count]]
            if not all(math.isfinite(value) for value in values):
                raise FixtureError(f"non-finite value in {name}")
        else:
            values = [int(item) for item in lines[line : line + count]]
        line += count
        if name in arrays:
            raise FixtureError(f"duplicate probe array {name}")
        arrays[name] = {"kind": kind.lower(), "shape": shape, "values": values}
    return {"mode": mode, "arrays": arrays}


def run_probe(probe: Path, mode: str, input_path: Path) -> tuple[dict[str, Any], bytes]:
    """Run one deterministic probe mode and parse its lossless output."""
    result = subprocess.run(
        [str(probe), mode, str(input_path)],
        check=True,
        capture_output=True,
        env=probe_environment(),
    )
    parsed = parse_probe_output(result.stdout)
    if parsed["mode"] != mode:
        raise FixtureError(f"probe returned {parsed['mode']} for requested mode {mode}")
    return parsed, result.stdout


def array(document: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one named array record from a parsed probe document."""
    try:
        return document["arrays"][name]
    except KeyError as exc:
        raise FixtureError(f"missing array {name}") from exc


def scalar(document: dict[str, Any], name: str) -> float:
    """Return a named scalar after validating its array representation."""
    record = array(document, name)
    if record["shape"] or len(record["values"]) != 1:
        raise FixtureError(f"{name} is not scalar")
    return float(record["values"][0])


def flat_index(shape: list[int], indices: tuple[int, ...]) -> int:
    """Return a Fortran-column-major flat index."""
    if len(shape) != len(indices):
        raise FixtureError("index rank mismatch")
    offset = 0
    stride = 1
    for dimension, index in zip(shape, indices, strict=True):
        if index < 0 or index >= dimension:
            raise FixtureError("array index out of range")
        offset += index * stride
        stride *= dimension
    return offset


def element(document: dict[str, Any], name: str, indices: tuple[int, ...]) -> float:
    """Return one Fortran-ordered array element as binary64."""
    record = array(document, name)
    return float(record["values"][flat_index(record["shape"], indices)])


def parse_input(path: Path) -> InputDocument:
    """Parse the deliberately narrow standalone-probe input grammar."""
    tokens = path.read_text(encoding="ascii").split()
    cursor = 0

    def take_int() -> int:
        nonlocal cursor
        value = int(tokens[cursor])
        cursor += 1
        return value

    def take_float() -> float:
        nonlocal cursor
        value = float(tokens[cursor])
        cursor += 1
        return value

    nat = take_int()
    numbers = [take_int() for _ in range(nat)]
    positions = [[take_float() for _ in range(3)] for _ in range(nat)]
    lattice_columns = [[take_float() for _ in range(3)] for _ in range(3)]
    nsh = take_int()
    qsh = [take_float() for _ in range(nsh)]
    qat = [take_float() for _ in range(nat)]
    dpat = [[take_float() for _ in range(3)] for _ in range(nat)]
    qpat = [[take_float() for _ in range(6)] for _ in range(nat)]
    if cursor != len(tokens):
        raise FixtureError("fixture input has trailing tokens")
    return {
        "atomic_numbers": numbers,
        "positions": positions,
        "lattice_columns": lattice_columns,
        "qsh": qsh,
        "qat": qat,
        "dpat": dpat,
        "qpat": qpat,
    }


def write_input(path: Path, data: InputDocument) -> None:
    """Write one probe input with round-trip-safe binary64 text."""
    lines = [
        str(len(data["atomic_numbers"])),
        " ".join(map(str, data["atomic_numbers"])),
    ]
    lines.extend(
        " ".join(f"{value:.17e}" for value in values) for values in data["positions"]
    )
    lines.extend(
        " ".join(f"{value:.17e}" for value in values)
        for values in data["lattice_columns"]
    )
    lines.append(str(len(data["qsh"])))
    lines.append(" ".join(f"{value:.17e}" for value in data["qsh"]))
    lines.append(" ".join(f"{value:.17e}" for value in data["qat"]))
    lines.extend(
        " ".join(f"{value:.17e}" for value in values) for values in data["dpat"]
    )
    lines.extend(
        " ".join(f"{value:.17e}" for value in values) for values in data["qpat"]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def coordinate_variant(data: InputDocument, delta: float) -> InputDocument:
    """Displace the selected Cartesian coordinate by ``delta`` bohr."""
    result = copy.deepcopy(data)
    result["positions"][COORDINATE_ATOM][COORDINATE_AXIS] += delta
    return result


def strain_variant(data: InputDocument, delta: float) -> InputDocument:
    """Apply the selected affine strain to positions and direct lattice."""
    result = copy.deepcopy(data)
    scale = 1.0 + delta
    for position in result["positions"]:
        position[STRAIN_ROW] *= scale
    for lattice_vector in result["lattice_columns"]:
        lattice_vector[STRAIN_ROW] *= scale
    return result


def variable_variant(
    data: InputDocument, variable: str, index: tuple[int, int], delta: float
) -> InputDocument:
    """Perturb one q/d/Q variable used for potential finite differences."""
    result = copy.deepcopy(data)
    atom, component = index
    if variable == "qat":
        result["qat"][atom] += delta
    elif variable == "dpat":
        result["dpat"][atom][component] += delta
    elif variable == "qpat":
        result["qpat"][atom][component] += delta
    else:
        raise FixtureError(f"unknown variable {variable}")
    return result


def selected_value(
    document: dict[str, Any], name: str, indices: tuple[int, ...] | None
) -> float:
    """Select either a scalar or indexed array value from probe output."""
    return (
        scalar(document, name) if indices is None else element(document, name, indices)
    )


def finite_difference(
    probe: Path,
    mode: str,
    base_input: InputDocument,
    value_name: str,
    value_indices: tuple[int, ...] | None,
    variant: Variant,
    analytic: float | None,
) -> dict[str, Any]:
    """Collect two-step central finite-difference evidence for one value."""
    samples = []
    with tempfile.TemporaryDirectory(prefix="xtbloom-periodic-terms-") as temp_dir:
        temp = Path(temp_dir)
        for step in FD_STEPS:
            plus_input = temp / "plus.in"
            minus_input = temp / "minus.in"
            write_input(plus_input, variant(base_input, step))
            write_input(minus_input, variant(base_input, -step))
            plus_doc, plus_raw = run_probe(probe, mode, plus_input)
            minus_doc, minus_raw = run_probe(probe, mode, minus_input)
            plus = selected_value(plus_doc, value_name, value_indices)
            minus = selected_value(minus_doc, value_name, value_indices)
            estimate = (plus - minus) / (2.0 * step)
            sample = {
                "step": step,
                "plus": plus,
                "minus": minus,
                "estimate": estimate,
                "plus_raw_sha256": sha256_bytes(plus_raw),
                "minus_raw_sha256": sha256_bytes(minus_raw),
            }
            if analytic is not None:
                sample["absolute_error"] = abs(estimate - analytic)
            samples.append(sample)
    result: dict[str, Any] = {
        "value": value_name,
        "indices_zero_based": list(value_indices) if value_indices is not None else [],
        "samples": samples,
    }
    if analytic is not None:
        result["analytic"] = analytic
    return result


def build_fd_evidence(
    probe: Path, mode: str, input_data: InputDocument, primary: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build the derivative evidence required for one isolated term mode."""
    evidence: list[dict[str, Any]] = []
    if mode == "cn":
        evidence.append(
            finite_difference(
                probe,
                mode,
                input_data,
                "gfn2_dexp_cn",
                (0,),
                coordinate_variant,
                element(
                    primary, "gfn2_dexp_dcndr", (COORDINATE_AXIS, COORDINATE_ATOM, 0)
                ),
            )
        )
        evidence[-1]["kind"] = "cartesian"
        evidence.append(
            finite_difference(
                probe,
                mode,
                input_data,
                "gfn2_dexp_cn",
                (0,),
                strain_variant,
                element(primary, "gfn2_dexp_dcndL", (STRAIN_ROW, STRAIN_COLUMN, 0)),
            )
        )
        evidence[-1]["kind"] = "affine_strain"
    elif mode == "integrals-h0":
        for name in ("overlap_matrix", "h0_matrix_hartree"):
            evidence.append(
                finite_difference(
                    probe, mode, input_data, name, (0, 4), coordinate_variant, None
                )
            )
            evidence[-1]["kind"] = "cartesian_convergence"
    else:
        energy = (
            "full_total_energy_hartree"
            if mode == "multipoles"
            else "total_energy_hartree"
        )
        gradient = (
            "full_gradient_hartree_per_bohr"
            if mode == "multipoles"
            else (
                "total_gradient_hartree_per_bohr"
                if mode == "d4"
                else "gradient_hartree_per_bohr"
            )
        )
        sigma = (
            "full_strain_derivatives_hartree"
            if mode == "multipoles"
            else (
                "total_strain_derivatives_hartree"
                if mode == "d4"
                else "strain_derivatives_hartree"
            )
        )
        evidence.append(
            finite_difference(
                probe,
                mode,
                input_data,
                energy,
                None,
                coordinate_variant,
                element(primary, gradient, (COORDINATE_AXIS, COORDINATE_ATOM)),
            )
        )
        evidence[-1]["kind"] = "cartesian"
        evidence.append(
            finite_difference(
                probe,
                mode,
                input_data,
                energy,
                None,
                strain_variant,
                element(primary, sigma, (STRAIN_ROW, STRAIN_COLUMN)),
            )
        )
        evidence[-1]["kind"] = "affine_strain"

    if mode == "d4":
        evidence.append(
            finite_difference(
                probe,
                mode,
                input_data,
                "d4_cn",
                (0,),
                coordinate_variant,
                element(primary, "d4_dcndr", (COORDINATE_AXIS, COORDINATE_ATOM, 0)),
            )
        )
        evidence[-1]["kind"] = "d4_cn_cartesian"
        evidence.append(
            finite_difference(
                probe,
                mode,
                input_data,
                "total_energy_hartree",
                None,
                lambda data, delta: variable_variant(data, "qat", (1, 0), delta),
                element(primary, "charge_potential_hartree_per_e", (1, 0)),
            )
        )
        evidence[-1]["kind"] = "atomic_charge_potential"
    elif mode == "multipoles":
        for variable, component, potential_name, potential_indices in (
            ("qat", 0, "full_charge_potential_hartree_per_e", (1, 0)),
            ("dpat", 0, "full_dipole_potential_hartree_per_e_bohr", (0, 1, 0)),
            ("qpat", 0, "full_quadrupole_potential_hartree_per_e_bohr2", (0, 1, 0)),
        ):
            evidence.append(
                finite_difference(
                    probe,
                    mode,
                    input_data,
                    "full_total_energy_hartree",
                    None,
                    lambda data, delta, variable=variable, component=component: (
                        variable_variant(data, variable, (1, component), delta)
                    ),
                    element(primary, potential_name, potential_indices),
                )
            )
            evidence[-1]["kind"] = f"{variable}_potential"
    return evidence


def runtime_attestation(probe: Path) -> dict[str, Any]:
    """Attest the probe build inputs and non-system runtime closure."""
    environment_root, build_root = oracle_build_paths()
    compiler = environment_root / "bin/x86_64-conda-linux-gnu-gfortran"
    version = subprocess.run(
        [str(compiler), "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    ldd = subprocess.run(
        ["ldd", str(probe)],
        check=True,
        capture_output=True,
        env=probe_environment(),
        text=True,
    ).stdout
    libraries = []
    for line in ldd.splitlines():
        tokens = line.strip().split()
        if len(tokens) < 3 or tokens[1] != "=>" or not tokens[2].startswith("/"):
            continue
        soname = tokens[0]
        candidate = Path(tokens[2]).resolve()
        if not candidate.is_file() or any(
            candidate.is_relative_to(system_root)
            for system_root in (Path("/lib"), Path("/usr/lib"))
        ):
            continue
        libraries.append(
            {
                "filename": candidate.name,
                "sha256": sha256_file(candidate),
                "soname": soname,
            }
        )
    libraries.sort(key=lambda item: item["soname"])
    if not any(item["soname"].startswith("libtblite.so") for item in libraries):
        raise FixtureError("probe runtime closure does not contain libtblite")
    module_dir = build_root / "libtblite.so.0.7.0.p"
    module_names = (
        "mctc_env.mod",
        "mctc_io.mod",
        "tblite_adjlist.mod",
        "tblite_basis_type.mod",
        "tblite_container_cache.mod",
        "tblite_coulomb_cache.mod",
        "tblite_cutoff.mod",
        "tblite_disp_cache.mod",
        "tblite_scf_potential.mod",
        "tblite_wavefunction_type.mod",
        "tblite_xtb_calculator.mod",
        "tblite_xtb_gfn2.mod",
        "tblite_xtb_h0.mod",
    )
    modules = {}
    for name in module_names:
        candidates = (
            module_dir / name,
            environment_root / "include/mctc-lib/modules" / name,
        )
        found = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if found is None:
            raise FixtureError(f"could not locate imported module {name}")
        modules[name] = sha256_file(found)
    return {
        "build_attestation": BUILD_ATTESTATION,
        "compiler": {"version": version, "sha256": sha256_file(compiler)},
        "runtime": {
            "discovery": "ldd",
            "non_system_libraries": libraries,
            "sha256": compact_json_sha256(libraries),
        },
        "fortran_modules": dict(sorted(modules.items())),
        "environment": {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    }


def generate(probe: Path, output: Path) -> None:
    """Generate raw, canonical, finite-difference, and manifest evidence."""
    input_path = corpus_root() / "inputs/water_skew.in"
    input_data = parse_input(input_path)
    raw_dir = output / "raw"
    golden_dir = output / "golden"
    raw_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in MODES:
        primary, raw = run_probe(probe, mode, input_path)
        raw_path = raw_dir / f"{mode}.txt"
        raw_path.write_bytes(raw)
        golden = {
            "schema": SCHEMA,
            "mode": mode,
            "input_id": INPUT_ID,
            "storage_order": "fortran-column-major",
            "arrays": primary["arrays"],
            "finite_difference": build_fd_evidence(probe, mode, input_data, primary),
        }
        golden_path = golden_dir / f"{mode}.json"
        golden_path.write_bytes(canonical_json(golden))
        records.append(
            {
                "mode": mode,
                "raw": f"data/conformance/periodic/terms/raw/{mode}.txt",
                "raw_sha256": sha256_file(raw_path),
                "golden": f"data/conformance/periodic/terms/golden/{mode}.json",
                "golden_sha256": sha256_file(golden_path),
            }
        )

    tool_dir = repository_root() / "tools/oracle/periodic_gfn2_terms"
    manifest = {
        "schema": "xtbloom-periodic-gfn2-term-manifest-v1",
        "generator": {
            "path": "tools/oracle/periodic_gfn2_terms/periodic_gfn2_terms.py",
            "sha256": sha256_file(tool_dir / "periodic_gfn2_terms.py"),
        },
        "input": {
            "id": INPUT_ID,
            "path": "data/conformance/periodic/terms/inputs/water_skew.in",
            "sha256": sha256_file(input_path),
            "units": {
                "positions": "bohr",
                "lattice_vectors": "bohr",
                "charges": "elementary_charge",
            },
            "lattice_order": "three direct lattice vectors stored as columns",
            "quadrupole_packing": ["xx", "xy", "yy", "xz", "yz", "zz"],
        },
        "probe": {
            "source": "tools/oracle/periodic_gfn2_terms/probe.f90",
            "source_sha256": sha256_file(tool_dir / "probe.f90"),
            "build_script": "tools/oracle/periodic_gfn2_terms/build_probe.sh",
            "build_script_sha256": sha256_file(tool_dir / "build_probe.sh"),
            "command_template": [
                "{probe}",
                "{mode}",
                "data/conformance/periodic/terms/inputs/water_skew.in",
            ],
            **runtime_attestation(probe),
        },
        "sources": SOURCE_RECORDS,
        "conventions": {
            "real_values": "IEEE binary64 atomic units",
            "array_storage": (
                "Fortran column-major; logical shapes are stored with every array"
            ),
            "cn_cartesian_derivative": (
                "[cartesian_axis, displaced_atom, coordination_atom]"
            ),
            "cn_strain_derivative": ("[affine_row, affine_column, coordination_atom]"),
            "gradient": "positive Cartesian derivative dE/dR in Hartree/bohr",
            "strain_derivative": (
                "dE/d_epsilon for simultaneous affine deformation of positions "
                "and direct lattice"
            ),
            "ao_order": "real spherical harmonics m=-l,...,+l; p order is py,pz,px",
            "quadrupole_packing": ["xx", "xy", "yy", "xz", "yz", "zz"],
            "moment_origin": (
                "multipole integral operator centered on the last AO index"
            ),
            "charge_ewald": (
                "neutral shell-resolved GFN2 ES2; no uniform charged-cell background"
            ),
            "multipole_boundary": (
                "conducting periodic boundary; reviewed direct cutoff is 100 bohr"
            ),
        },
        "legal": {
            "probe_license": "GPL-3.0-or-later",
            "oracle_license": "LGPL-3.0-or-later",
            "source_data_boundary": (
                "Tracked raw and golden files are independently generated numerical "
                "output; no upstream source or geometry bytes are copied."
            ),
        },
        "fixtures": records,
        "tolerances": {
            "absolute_scalar": 5.0e-12,
            "matrix_symmetry": 5.0e-12,
            "net_gradient": 5.0e-11,
            "finite_difference": 2.0e-7,
            "finite_difference_cn": 2.0e-7,
            "finite_difference_convergence": 2.0e-6,
            "quadrupole_trace": 5.0e-12,
        },
    }
    (output / "manifest.json").write_bytes(canonical_json(manifest))


def assert_close(
    actual: float, expected: float, tolerance: float, message: str
) -> None:
    """Raise ``FixtureError`` unless two scalar values agree absolutely."""
    if abs(actual - expected) > tolerance:
        raise FixtureError(
            f"{message}: {actual:.17e} vs {expected:.17e}, tolerance {tolerance:.3e}"
        )


def matrix_value(record: dict[str, Any], row: int, column: int) -> float:
    """Return one element from a rank-two Fortran-ordered record."""
    return float(record["values"][flat_index(record["shape"], (row, column))])


def check_symmetric(record: dict[str, Any], tolerance: float, name: str) -> None:
    """Require a square matrix record to be symmetric within tolerance."""
    if len(record["shape"]) != 2 or record["shape"][0] != record["shape"][1]:
        raise FixtureError(f"{name} is not square")
    size = record["shape"][0]
    for row in range(size):
        for column in range(size):
            assert_close(
                matrix_value(record, row, column),
                matrix_value(record, column, row),
                tolerance,
                f"{name} symmetry",
            )


def sum_array(document: dict[str, Any], name: str) -> float:
    """Accurately sum all values in a named array record."""
    return math.fsum(float(value) for value in array(document, name)["values"])


def check_net_gradient(document: dict[str, Any], name: str, tolerance: float) -> None:
    """Require every Cartesian component of a gradient to sum to zero."""
    record = array(document, name)
    if len(record["shape"]) != 2 or record["shape"][0] != 3:
        raise FixtureError(f"{name} has the wrong gradient shape")
    for axis in range(3):
        total = math.fsum(
            float(record["values"][flat_index(record["shape"], (axis, atom))])
            for atom in range(record["shape"][1])
        )
        assert_close(total, 0.0, tolerance, f"{name} net axis {axis}")


def check_finite_differences(
    document: dict[str, Any], tolerances: dict[str, float]
) -> None:
    """Recompute and validate every retained finite-difference estimate."""
    for evidence in document["finite_difference"]:
        estimates = []
        for sample in evidence["samples"]:
            recomputed = (sample["plus"] - sample["minus"]) / (2.0 * sample["step"])
            assert_close(
                recomputed, sample["estimate"], 5.0e-13, "stored finite difference"
            )
            estimates.append(sample["estimate"])
            if "analytic" in evidence:
                tolerance = (
                    tolerances["finite_difference_cn"]
                    if "cn" in evidence["kind"]
                    else tolerances["finite_difference"]
                )
                assert_close(
                    sample["estimate"],
                    evidence["analytic"],
                    tolerance,
                    evidence["kind"],
                )
                assert_close(
                    sample["absolute_error"],
                    abs(sample["estimate"] - evidence["analytic"]),
                    5.0e-13,
                    "FD error",
                )
        assert_close(
            estimates[0],
            estimates[1],
            tolerances["finite_difference_convergence"],
            f"{evidence['kind']} step convergence",
        )
        if "analytic" not in evidence and abs(estimates[-1]) < 1.0e-8:
            raise FixtureError(
                f"{evidence['kind']} does not exercise a responsive matrix element"
            )


def check_invariants(
    documents: dict[str, dict[str, Any]], tolerances: dict[str, float]
) -> None:
    """Check cross-array scientific invariants for all six term families."""
    cn = documents["cn"]
    derivative = array(cn, "gfn2_dexp_dcndr")
    for axis in range(3):
        for target in range(derivative["shape"][2]):
            total = math.fsum(
                derivative["values"][
                    flat_index(derivative["shape"], (axis, atom, target))
                ]
                for atom in range(derivative["shape"][1])
            )
            assert_close(
                total, 0.0, tolerances["absolute_scalar"], "CN translation derivative"
            )

    repulsion = documents["repulsion"]
    assert_close(
        sum_array(repulsion, "per_atom_energy_hartree"),
        scalar(repulsion, "total_energy_hartree"),
        tolerances["absolute_scalar"],
        "repulsion per-atom recomposition",
    )
    check_net_gradient(
        repulsion, "gradient_hartree_per_bohr", tolerances["net_gradient"]
    )

    d4 = documents["d4"]
    assert_close(
        sum_array(d4, "nonsc_per_atom_energy_hartree")
        + sum_array(d4, "pair_per_atom_energy_hartree"),
        scalar(d4, "total_energy_hartree"),
        tolerances["absolute_scalar"],
        "D4 energy decomposition",
    )
    check_net_gradient(
        d4, "total_gradient_hartree_per_bohr", tolerances["net_gradient"]
    )

    h0 = documents["integrals-h0"]
    check_symmetric(
        array(h0, "overlap_matrix"), tolerances["matrix_symmetry"], "overlap"
    )
    check_symmetric(array(h0, "h0_matrix_hartree"), tolerances["matrix_symmetry"], "H0")
    quadrupole = array(h0, "quadrupole_matrices_bohr2")
    nao = quadrupole["shape"][1]
    for row in range(nao):
        for column in range(nao):
            trace = math.fsum(
                quadrupole["values"][
                    flat_index(quadrupole["shape"], (component, row, column))
                ]
                for component in (0, 2, 5)
            )
            assert_close(trace, 0.0, tolerances["quadrupole_trace"], "quadrupole trace")

    charge = documents["charge-ewald"]
    amat = array(charge, "shell_coulomb_matrix_hartree_per_e2")
    check_symmetric(amat, tolerances["matrix_symmetry"], "shell Coulomb matrix")
    charges = array(charge, "fixed_shell_charges_e")["values"]
    potential = array(charge, "shell_charge_potential_hartree_per_e")
    contracted = []
    for row in range(len(charges)):
        contracted.append(
            math.fsum(
                matrix_value(amat, row, column) * charges[column]
                for column in range(len(charges))
            )
        )
        assert_close(
            contracted[-1],
            potential["values"][row],
            tolerances["absolute_scalar"],
            "shell potential",
        )
    assert_close(
        0.5 * math.fsum(q * v for q, v in zip(charges, contracted, strict=True)),
        scalar(charge, "total_energy_hartree"),
        tolerances["absolute_scalar"],
        "shell Ewald matrix contraction",
    )
    check_net_gradient(charge, "gradient_hartree_per_bohr", tolerances["net_gradient"])

    multipoles = documents["multipoles"]
    for state in ("dipole_only", "charge_quadrupole", "full"):
        assert_close(
            sum_array(multipoles, f"{state}_per_atom_total_energy_hartree"),
            scalar(multipoles, f"{state}_total_energy_hartree"),
            tolerances["absolute_scalar"],
            f"{state} per-atom energy",
        )
        assert_close(
            sum_array(multipoles, f"{state}_per_atom_aes_energy_hartree")
            + sum_array(multipoles, f"{state}_per_atom_axc_energy_hartree"),
            scalar(multipoles, f"{state}_total_energy_hartree"),
            tolerances["absolute_scalar"],
            f"{state} AES+AXC",
        )
        check_net_gradient(
            multipoles, f"{state}_gradient_hartree_per_bohr", tolerances["net_gradient"]
        )

    for document in documents.values():
        check_finite_differences(document, tolerances)


def check(root: Path | None = None) -> None:
    """Verify the committed corpus and all scientific evidence offline."""
    root = root or corpus_root()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "xtbloom-periodic-gfn2-term-manifest-v1":
        raise FixtureError("term manifest has the wrong schema")
    repo = repository_root()
    generator = manifest.get("generator")
    if not isinstance(generator, dict) or sha256_file(
        repo / str(generator.get("path", ""))
    ) != generator.get("sha256"):
        raise FixtureError("generator hash mismatch")
    probe = manifest["probe"]
    if probe.get("build_attestation") != BUILD_ATTESTATION:
        raise FixtureError("tblite build attestation identity mismatch")
    runtime = probe.get("runtime")
    libraries = (
        runtime.get("non_system_libraries") if isinstance(runtime, dict) else None
    )
    if (
        not isinstance(libraries, list)
        or not libraries
        or runtime.get("sha256") != compact_json_sha256(libraries)
    ):
        raise FixtureError("probe non-system runtime closure is invalid")
    if manifest.get("sources") != SOURCE_RECORDS:
        raise FixtureError("upstream source identity mismatch")
    for key, hash_key in (
        ("source", "source_sha256"),
        ("build_script", "build_script_sha256"),
    ):
        if sha256_file(repo / probe[key]) != probe[hash_key]:
            raise FixtureError(f"{key} hash mismatch")
    input_record = manifest["input"]
    input_path = repo / input_record["path"]
    if sha256_file(input_path) != input_record["sha256"]:
        raise FixtureError("term input hash mismatch")
    parsed_input = parse_input(input_path)
    if (
        abs(math.fsum(parsed_input["qat"])) > 1.0e-14
        or abs(math.fsum(parsed_input["qsh"])) > 1.0e-14
    ):
        raise FixtureError("primary charge fixtures must be neutral")
    for quadrupole in parsed_input["qpat"]:
        assert_close(
            quadrupole[0] + quadrupole[2] + quadrupole[5],
            0.0,
            1.0e-14,
            "input quadrupole trace",
        )

    documents = {}
    seen_modes = set()
    for fixture in manifest["fixtures"]:
        mode = fixture["mode"]
        if mode in seen_modes:
            raise FixtureError(f"duplicate fixture {mode}")
        seen_modes.add(mode)
        raw_path = repo / fixture["raw"]
        golden_path = repo / fixture["golden"]
        if sha256_file(raw_path) != fixture["raw_sha256"]:
            raise FixtureError(f"raw hash mismatch for {mode}")
        if sha256_file(golden_path) != fixture["golden_sha256"]:
            raise FixtureError(f"golden hash mismatch for {mode}")
        raw_document = parse_probe_output(raw_path.read_bytes())
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        if golden.get("schema") != SCHEMA or golden.get("mode") != mode:
            raise FixtureError(f"golden identity mismatch for {mode}")
        if golden["arrays"] != raw_document["arrays"]:
            raise FixtureError(f"raw/golden numerical mismatch for {mode}")
        documents[mode] = golden
    if seen_modes != set(MODES):
        raise FixtureError(f"missing term modes: {sorted(set(MODES) - seen_modes)}")
    check_invariants(documents, manifest["tolerances"])


def compare(probe: Path) -> None:
    """Regenerate with the live oracle and byte-compare every artifact."""
    with tempfile.TemporaryDirectory(
        prefix="xtbloom-periodic-terms-compare-"
    ) as temp_dir:
        output = Path(temp_dir) / "terms"
        generate(probe, output)
        expected = corpus_root()
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(output)
            expected_path = expected / relative
            if (
                not expected_path.is_file()
                or path.read_bytes() != expected_path.read_bytes()
            ):
                raise FixtureError(
                    f"live oracle differs from committed fixture: {relative}"
                )


def main() -> int:
    """Run the requested generate, offline-check, or live-compare command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="verify the committed corpus offline")
    generate_parser = subparsers.add_parser(
        "generate", help="generate fixtures from the pinned probe"
    )
    generate_parser.add_argument("--probe", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, default=corpus_root())
    compare_parser = subparsers.add_parser(
        "compare", help="regenerate and byte-compare with committed fixtures"
    )
    compare_parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            check()
        elif args.command == "generate":
            generate(args.probe.resolve(), args.output_dir.resolve())
        else:
            compare(args.probe.resolve())
    except (
        FixtureError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        KeyError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
