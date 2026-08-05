#!/usr/bin/env python3
"""Audit-ready public-C-ABI FRESH/WARM latency sweep by atom count.

The runner intentionally depends only on gpuxtb's committed ctypes conformance
adapter and the Python standard library.  One context, batch descriptor, compute
options image, and set of caller-owned result buffers remain alive for an
entire cell.  A WARM cell performs exactly one untimed FRESH call to publish the
strict checkpoint before any warmup or measured WARM call.

Artifacts are never discovered by scanning an output directory.  Both output
paths are explicit and existing files are rejected unless the caller opts in to
replacement, preventing stale FRESH/WARM or cross-machine rows from being
silently combined.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import io
import json
import math
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_TOOLS = REPOSITORY_ROOT / "tools" / "conformance"
if str(CONFORMANCE_TOOLS) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_TOOLS))
public_api = importlib.import_module("gpuxtb_public_api")

SCHEMA_VERSION = 1
ANGSTROM_TO_BOHR = 1.8897261254579021
DEFAULT_NATOMS = (32, 62, 98, 122)
CONFORMANCE_MANIFEST = REPOSITORY_ROOT / "data" / "conformance" / "manifest.json"
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
    "MKL_INTERFACE_LAYER",
    "MKL_THREADING_LAYER",
)


def _manifest_tolerance_defaults() -> tuple[float, float, float, float, dict[str, Any]]:
    """Read the committed primary and live cross-engine conformance gates."""
    manifest_bytes = CONFORMANCE_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    tolerances = manifest["cross_engine_tolerances"]
    energy = float(tolerances["energy"]["atol"])
    force = float(tolerances["forces"]["atol"])
    primary_energy = float(manifest["tolerances"]["energy"]["atol"])
    primary_force = float(manifest["tolerances"]["forces"]["atol"])
    return (
        energy,
        force,
        primary_energy,
        primary_force,
        {
            "path": str(CONFORMANCE_MANIFEST),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "json_field": "cross_engine_tolerances",
        },
    )


(
    DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
    DEFAULT_CROSS_ENGINE_FORCE_ATOL,
    DEFAULT_ENERGY_ATOL_LIMIT,
    DEFAULT_FORCE_ATOL,
    CROSS_ENGINE_TOLERANCE_SOURCE,
) = _manifest_tolerance_defaults()
PRIMARY_FORCE_TOLERANCE_SOURCE = {
    **CROSS_ENGINE_TOLERANCE_SOURCE,
    "json_field": "tolerances.forces",
}
PRIMARY_ENERGY_TOLERANCE_SOURCE = {
    **CROSS_ENGINE_TOLERANCE_SOURCE,
    "json_field": "tolerances.energy",
}
GPUXTB_CONFORMANCE_MAX_SCC_ITERATIONS = 500
GPUXTB_CONFORMANCE_CHARGE_TOLERANCE = 1.0e-10
GPUXTB_CONFORMANCE_ENERGY_TOLERANCE = 1.0e-12
GPUXTB_CONFORMANCE_ELECTRONIC_TEMPERATURE = 300.0 * public_api.GPUXTB_KELVIN_TO_HARTREE


class BenchmarkError(RuntimeError):
    """An invalid benchmark request or failed public inference."""


class Runner(Protocol):
    """Minimal persistent runner surface used by the hardware-free tests."""

    compute_options: dict[str, Any]

    def set_start_mode(self, mode: str) -> None: ...

    def invoke(self) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Molecule:
    """One immutable neutral closed-shell molecule in public ABI units."""

    name: str
    atomic_numbers: tuple[int, ...]
    positions_bohr: tuple[float, ...]

    @property
    def natoms(self) -> int:
        return len(self.atomic_numbers)


@dataclass(frozen=True)
class Cell:
    """One independently reviewable natoms/batch benchmark coordinate."""

    engine: str
    molecule: Molecule
    batch_size: int
    backend: str
    property_name: str


@dataclass(frozen=True)
class Protocol:
    """Timing and correctness settings shared by every cell in one run."""

    start_mode: str
    warmups: int
    repetitions: int
    energy_atol_hartree: float
    force_atol_hartree_per_bohr: float


@dataclass(frozen=True)
class ReferenceRow:
    """One validated FRESH result selected by complete workload identity."""

    energies_hartree: tuple[float, ...]
    forces_hartree_per_bohr: tuple[float, ...] | None
    compute_options: dict[str, Any]


@dataclass(frozen=True)
class ReferenceArtifact:
    """Parsed reference bytes and their matching digest, retained in memory."""

    path: Path
    sha256: str
    run_identity: dict[str, Any]
    rows: dict[str, ReferenceRow]


@dataclass
class BatchStorage:
    """Duck-typed storage accepted by ``gpuxtb_public_api._make_batch``."""

    atom_offsets: list[int]
    atomic_numbers: list[int]
    positions: list[float]
    molecular_charges: list[float]
    unpaired_electrons: list[int]
    spin_channels: list[int]
    point_charge_offsets: list[int]
    point_charge_positions: list[float]
    point_charge_values: list[float]
    point_charge_gammas: list[float]
    slices: list[SystemSlice]
    keepalive: list[Any]


def configure_gpuxtb_conformance_scc(options: Any) -> None:
    """Pin the SCC convergence controls used by the conformance oracle.

    ``gpuxtb_compute_options_init`` already supplies the public 300 K default,
    which is deliberately retained rather than overwritten here.
    """
    options.max_scc_iterations = GPUXTB_CONFORMANCE_MAX_SCC_ITERATIONS
    options.charge_tolerance = GPUXTB_CONFORMANCE_CHARGE_TOLERANCE
    options.energy_tolerance = GPUXTB_CONFORMANCE_ENERGY_TOLERANCE


@dataclass(frozen=True)
class SystemSlice:
    """Atom/point offsets consumed by the persistent reference adapters."""

    atom_begin: int
    atom_end: int
    point_begin: int = 0
    point_end: int = 0


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse a nonempty comma-separated list of positive integers."""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("selection must contain positive integers")
    return values


def sha256_file(path: Path) -> str:
    """Hash one required regular file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_numbers(values: Sequence[int | float], format_code: str) -> str:
    """Hash a numeric sequence using a portable little-endian binary image."""
    digest = hashlib.sha256()
    packer = struct.Struct("<" + format_code)
    for value in values:
        digest.update(packer.pack(value))
    return digest.hexdigest()


def run_text(command: Sequence[str]) -> str | None:
    """Return stdout from a bounded read-only diagnostic command."""
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_state(path: Path) -> dict[str, Any]:
    """Capture exact revision and dirty state for the measured checkout."""
    revision = run_text(("git", "-C", str(path), "rev-parse", "HEAD"))
    porcelain = run_text(("git", "-C", str(path), "status", "--porcelain"))
    return {
        "path": str(path.resolve()),
        "revision": revision,
        "dirty": None if revision is None else bool(porcelain),
    }


def cpu_model() -> str | None:
    """Read the first Linux CPU model name when available."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def process_affinity() -> list[int] | None:
    """Return the exact logical CPUs available to the benchmark process."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return sorted(getter(0))
    except OSError:
        return None


def _file_identity(path: Path) -> dict[str, Any]:
    """Record one build input/provider path and its current content hash."""
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
        "is_file": resolved.is_file(),
    }


def _parse_cmake_cache(cache: Path) -> dict[str, str]:
    """Parse typed CMake cache assignments without losing configured flags."""
    entries: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        declaration, value = line.split("=", 1)
        if ":" not in declaration:
            continue
        name, _type_name = declaration.rsplit(":", 1)
        entries[name] = value
    return entries


def _cmake_build_metadata(library: Path, cache: Path) -> dict[str, Any]:
    """Capture compiler, flags, source revision, and BLAS/LAPACK provider."""
    entries = _parse_cmake_cache(cache)
    selected_names = (
        "BUILD_SHARED_LIBS",
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "CMAKE_EXE_LINKER_FLAGS",
        "CMAKE_SHARED_LINKER_FLAGS",
        "CMAKE_GENERATOR",
        "CMAKE_HOME_DIRECTORY",
        "GPUXTB_ENABLE_CUDA",
        "GPUXTB_MKL_RT_LIBRARY",
    )
    selected = {name: entries.get(name) for name in selected_names}
    source_path = Path(entries["CMAKE_HOME_DIRECTORY"]).resolve()
    compiler_text = entries.get("CMAKE_CXX_COMPILER")
    compiler_path = Path(compiler_text).resolve() if compiler_text else None
    provider_text = entries.get("GPUXTB_MKL_RT_LIBRARY")
    provider_path = Path(provider_text).resolve() if provider_text else None
    source_inputs = []
    for relative in ("CMakeLists.txt", "cmake/gpuxtb.map"):
        candidate = source_path / relative
        if candidate.is_file():
            source_inputs.append(_file_identity(candidate))
    return {
        "build_system": "cmake",
        "build_directory": str(library.parent.resolve()),
        "cmake_version": run_text(("cmake", "--version")),
        "cache": _file_identity(cache),
        "cache_entries": selected,
        "source": {
            "path": str(source_path),
            "git": git_state(source_path),
            "inputs": source_inputs,
        },
        "compiler": (
            {
                **_file_identity(compiler_path),
                "version": run_text((str(compiler_path), "--version")),
            }
            if compiler_path is not None
            else None
        ),
        "dependency_provider": (
            {
                "role": "LP64 LAPACKE+CBLAS runtime",
                **_file_identity(provider_path),
            }
            if provider_path is not None
            else None
        ),
    }


def _read_json_metadata(path: Path) -> tuple[Any, dict[str, Any]]:
    """Parse one Meson introspection file and retain its exact byte hash."""
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _meson_build_metadata(library: Path, info_directory: Path) -> dict[str, Any]:
    """Capture Meson configuration, compilers, dependencies, and source Git state."""
    names = (
        "meson-info.json",
        "intro-projectinfo.json",
        "intro-compilers.json",
        "intro-buildoptions.json",
        "intro-dependencies.json",
    )
    documents: dict[str, Any] = {}
    file_identities: dict[str, Any] = {}
    for name in names:
        document, identity = _read_json_metadata(info_directory / name)
        documents[name] = document
        file_identities[name] = identity
    meson_info = documents["meson-info.json"]
    source_path = Path(meson_info["directories"]["source"]).resolve()
    option_items = documents["intro-buildoptions.json"]
    option_map = {
        item["name"]: item.get("value")
        for item in option_items
        if isinstance(item, dict) and "name" in item
    }
    selected_options = {
        name: option_map.get(name)
        for name in (
            "backend",
            "buildtype",
            "optimization",
            "debug",
            "b_ndebug",
            "b_lto",
            "b_sanitize",
            "default_library",
            "c_args",
            "c_link_args",
            "fortran_args",
            "fortran_link_args",
        )
    }
    dependencies = []
    for dependency in documents["intro-dependencies.json"]:
        linked_files = []
        for argument in dependency.get("link_args", []):
            candidate = Path(argument)
            if candidate.is_absolute() and candidate.is_file():
                linked_files.append(_file_identity(candidate))
        dependencies.append(
            {
                "name": dependency.get("name"),
                "type": dependency.get("type"),
                "version": dependency.get("version"),
                "compile_args": dependency.get("compile_args"),
                "link_args": dependency.get("link_args"),
                "linked_files": linked_files,
            }
        )
    compilers: dict[str, Any] = {}
    for machine, languages in documents["intro-compilers.json"].items():
        compilers[machine] = {}
        for language, details in languages.items():
            executable_files = []
            for executable in details.get("exelist", []):
                resolved = shutil.which(executable)
                if resolved is not None:
                    executable_files.append(_file_identity(Path(resolved)))
            compilers[machine][language] = {
                **details,
                "executable_files": executable_files,
            }
    source_inputs = []
    for relative in ("meson.build", "meson_options.txt"):
        candidate = source_path / relative
        if candidate.is_file():
            source_inputs.append(_file_identity(candidate))
    return {
        "build_system": "meson",
        "build_directory": str(library.parent.resolve()),
        "meson_version": meson_info.get("meson_version"),
        "introspection_files": file_identities,
        "project": documents["intro-projectinfo.json"],
        "source": {
            "path": str(source_path),
            "git": git_state(source_path),
            "inputs": source_inputs,
        },
        "compilers": compilers,
        "build_options": selected_options,
        "dependencies": dependencies,
    }


def build_metadata(library: Path) -> dict[str, Any]:
    """Discover adjacent CMake or Meson provenance without guessing settings."""
    cache = library.parent / "CMakeCache.txt"
    if cache.is_file():
        try:
            return _cmake_build_metadata(library, cache)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise BenchmarkError(f"invalid adjacent CMake metadata: {exc}") from exc
    meson_info = library.parent / "meson-info"
    if all(
        (meson_info / name).is_file()
        for name in (
            "meson-info.json",
            "intro-projectinfo.json",
            "intro-compilers.json",
            "intro-buildoptions.json",
            "intro-dependencies.json",
        )
    ):
        try:
            return _meson_build_metadata(library, meson_info)
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise BenchmarkError(f"invalid adjacent Meson metadata: {exc}") from exc
    return {
        "build_system": "unavailable",
        "reason": "no adjacent CMakeCache.txt or complete meson-info introspection set",
    }


def exact_argv(arguments: Sequence[str]) -> list[str]:
    """Canonicalize the actual interpreter, script, and parsed CLI arguments."""
    return [sys.executable, str(Path(__file__).resolve()), *arguments]


def collect_run_identity(
    engine: str,
    library: Path,
    arguments: Sequence[str],
    reference_artifact: ReferenceArtifact | None,
) -> dict[str, Any]:
    """Collect the immutable evidence repeated at top-level and per row."""
    resolved_library = library.resolve()
    return {
        "argv": exact_argv(arguments),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": git_state(REPOSITORY_ROOT),
        "library": {
            "engine": engine,
            "path": str(resolved_library),
            "sha256": sha256_file(resolved_library),
            "build": build_metadata(resolved_library),
        },
        "runner": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "hardware": {
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "process_affinity": process_affinity(),
        },
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_NAMES
        },
        "fresh_reference_artifact": (
            {
                "path": str(reference_artifact.path),
                "sha256": reference_artifact.sha256,
            }
            if reference_artifact is not None
            else None
        ),
    }


def _is_sha256(value: Any) -> bool:
    """Return whether ``value`` is one lowercase or uppercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _is_git_revision(value: Any) -> bool:
    """Accept complete SHA-1 or SHA-256 Git object identifiers."""
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _gpuxtb_binary_source_identity(
    run_identity: dict[str, Any], context: str
) -> tuple[str, str]:
    """Validate one clean in-tree gpuxtb binary and return binary/source hashes."""
    repository = run_identity.get("repository")
    library = run_identity.get("library")
    if not isinstance(repository, dict) or not isinstance(library, dict):
        raise BenchmarkError(f"{context} lacks repository/library identity")
    revision = repository.get("revision")
    if not _is_git_revision(revision) or repository.get("dirty") is not False:
        raise BenchmarkError(f"{context} repository must be a clean Git revision")
    library_sha = library.get("sha256")
    if library.get("engine") != "gpuxtb" or not _is_sha256(library_sha):
        raise BenchmarkError(f"{context} lacks a valid gpuxtb library SHA-256")
    build = library.get("build")
    source_git = build.get("source", {}).get("git") if isinstance(build, dict) else None
    if (
        not isinstance(source_git, dict)
        or source_git.get("dirty") is not False
        or source_git.get("revision") != revision
    ):
        raise BenchmarkError(
            f"{context} gpuxtb binary source must be clean and match repository revision"
        )
    return library_sha, revision


def apply_current_evidence_policy(
    run_identity: dict[str, Any], allow_dirty_evidence: bool
) -> None:
    """Reject dirty benchmark sources, or label an explicit development override."""
    repository = run_identity.get("repository")
    if (
        not isinstance(repository, dict)
        or not _is_git_revision(repository.get("revision"))
        or repository.get("dirty") is None
    ):
        raise BenchmarkError("current benchmark repository Git state is unavailable")
    dirty_sources = []
    if repository["dirty"]:
        dirty_sources.append("benchmark repository")
    library = run_identity.get("library")
    build = library.get("build") if isinstance(library, dict) else None
    source_git = build.get("source", {}).get("git") if isinstance(build, dict) else None
    if isinstance(source_git, dict) and source_git.get("dirty"):
        dirty_sources.append("selected library source")
    if dirty_sources and not allow_dirty_evidence:
        raise BenchmarkError(
            "dirty sources cannot produce benchmark evidence: "
            + ", ".join(dirty_sources)
        )
    run_identity["evidence_eligibility"] = {
        "status": "development_only_dirty" if dirty_sources else "eligible",
        "allow_dirty_evidence": allow_dirty_evidence,
        "dirty_sources": dirty_sources,
    }


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise BenchmarkError("cannot normalize a zero molecular direction")
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _orthogonal(axis: Sequence[float]) -> tuple[float, float, float]:
    """Construct one deterministic unit vector perpendicular to ``axis``."""
    candidate = (
        (-axis[1], axis[0], 0.0) if abs(axis[0]) < 0.9 else (0.0, -axis[2], axis[1])
    )
    return _unit(candidate)


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    """Return the three-dimensional cross product."""
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def make_alkane(natoms: int) -> Molecule:
    """Build a deterministic all-trans-like C_n H_(2n+2) molecule.

    The accepted atom counts satisfy ``natoms = 3*ncarbon + 2``.  The geometry
    is a benchmark input rather than a generated scientific golden; its only
    role is to provide a stable, non-overlapping size sweep shared by FRESH and
    WARM calls.
    """
    if natoms < 5 or (natoms - 2) % 3 != 0:
        raise BenchmarkError(
            f"natoms={natoms} is not representable as an alkane C_nH_(2n+2)"
        )
    carbons = (natoms - 2) // 3
    cc = 1.54
    ch = 1.09
    half_external = 0.5 * math.acos(1.0 / 3.0)
    even_step = (cc * math.cos(half_external), cc * math.sin(half_external), 0.0)
    odd_step = (cc * math.cos(half_external), -cc * math.sin(half_external), 0.0)
    carbon_positions: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for index in range(1, carbons):
        step = even_step if index % 2 == 1 else odd_step
        previous = carbon_positions[-1]
        carbon_positions.append(tuple(previous[axis] + step[axis] for axis in range(3)))

    atoms: list[tuple[int, tuple[float, float, float]]] = []
    for index, position in enumerate(carbon_positions):
        neighbors = []
        if index > 0:
            neighbors.append(carbon_positions[index - 1])
        if index + 1 < carbons:
            neighbors.append(carbon_positions[index + 1])
        if not neighbors:
            directions = (
                _unit((1.0, 1.0, 1.0)),
                _unit((1.0, -1.0, -1.0)),
                _unit((-1.0, 1.0, -1.0)),
                _unit((-1.0, -1.0, 1.0)),
            )
        elif len(neighbors) == 1:
            # ``axis`` points away from the C-C bond.  The three C-H vectors
            # form a 120-degree ring around it and each has dot(+axis)=1/3,
            # giving the tetrahedral dot product -1/3 against the C-C bond.
            axis = _unit(tuple(position[i] - neighbors[0][i] for i in range(3)))
            perpendicular = _orthogonal(axis)
            binormal = _unit(_cross(axis, perpendicular))
            axial = 1.0 / 3.0
            radial = math.sqrt(8.0 / 9.0)
            directions = tuple(
                tuple(
                    axial * axis[component]
                    + radial
                    * (
                        math.cos(azimuth) * perpendicular[component]
                        + math.sin(azimuth) * binormal[component]
                    )
                    for component in range(3)
                )
                for azimuth in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
            )
        else:
            # Complete the tetrahedron around the two backbone bond vectors.
            # This also moves the hydrogens away from the carbon plane instead
            # of placing them directly above one another on adjacent carbons.
            first = _unit(tuple(neighbors[0][i] - position[i] for i in range(3)))
            second = _unit(tuple(neighbors[1][i] - position[i] for i in range(3)))
            bisector = _unit(tuple(first[i] + second[i] for i in range(3)))
            normal = _unit(_cross(first, second))
            axial = -1.0 / math.sqrt(3.0)
            radial = math.sqrt(2.0 / 3.0)
            directions = (
                tuple(axial * bisector[i] + radial * normal[i] for i in range(3)),
                tuple(axial * bisector[i] - radial * normal[i] for i in range(3)),
            )
        for direction in directions:
            atoms.append(
                (1, tuple(position[axis] + ch * direction[axis] for axis in range(3)))
            )
    atoms.extend((6, position) for position in carbon_positions)
    if len(atoms) != natoms:
        raise BenchmarkError(
            f"internal alkane builder produced {len(atoms)} atoms instead of {natoms}"
        )
    numbers = tuple(number for number, _ in atoms)
    positions = tuple(
        coordinate * ANGSTROM_TO_BOHR for _, point in atoms for coordinate in point
    )
    return Molecule(f"C{carbons}H{2 * carbons + 2}", numbers, positions)


def workload_identity(cell: Cell) -> dict[str, Any]:
    """Describe every input that selects a row across benchmark artifacts."""
    molecule = cell.molecule
    return {
        "molecule": molecule.name,
        "natoms": molecule.natoms,
        "batch_size": cell.batch_size,
        "backend": cell.backend,
        "memory_mode": "host",
        "property": cell.property_name,
        "atomic_numbers_sha256": _sha256_numbers(molecule.atomic_numbers, "q"),
        "positions_bohr_sha256": _sha256_numbers(molecule.positions_bohr, "d"),
        "position_units": "bohr",
        "molecular_charge_e": 0.0,
        "unpaired_electrons": 0,
        "spin_channels": 1,
    }


def workload_key(identity: dict[str, Any]) -> str:
    """Serialize a validated workload identity into one unambiguous key."""
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


def make_storage(molecule: Molecule, batch_size: int) -> BatchStorage:
    """Repeat one molecule into a persistent public ragged batch."""
    atom_offsets = [0]
    atomic_numbers: list[int] = []
    positions: list[float] = []
    slices: list[SystemSlice] = []
    for _ in range(batch_size):
        begin = len(atomic_numbers)
        atomic_numbers.extend(molecule.atomic_numbers)
        positions.extend(molecule.positions_bohr)
        atom_offsets.append(len(atomic_numbers))
        slices.append(SystemSlice(begin, len(atomic_numbers)))
    return BatchStorage(
        atom_offsets=atom_offsets,
        atomic_numbers=atomic_numbers,
        positions=positions,
        molecular_charges=[0.0] * batch_size,
        unpaired_electrons=[0] * batch_size,
        spin_channels=[1] * batch_size,
        point_charge_offsets=[0] * (batch_size + 1),
        point_charge_positions=[],
        point_charge_values=[],
        point_charge_gammas=[],
        slices=slices,
        keepalive=[],
    )


class GpuxtbRunner:
    """Own one persistent public-C-ABI context, descriptor, and result image."""

    def __init__(
        self,
        library_path: Path,
        cell: Cell,
        cpu_threads: int,
        device_id: int,
    ) -> None:
        self.library = public_api._configure_library(library_path)
        self.storage = make_storage(cell.molecule, cell.batch_size)
        self.context = public_api._make_context(
            self.library, cell.backend, device_id, cpu_threads
        )
        self.memory = public_api.DescriptorMemory("host", device_id)
        self.batch = public_api._make_batch(
            self.library,
            self.storage,
            self.memory,
            include_spin_channels=True,
        )
        self.options = public_api.ComputeOptions()
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_compute_options_init(
                ctypes.byref(self.options), ctypes.sizeof(self.options)
            ),
            "gpuxtb_compute_options_init",
        )
        self.options.model = public_api.GPUXTB_MODEL_GFN2_XTB
        self.options.flags = public_api.GPUXTB_COMPUTE_ENERGY
        if cell.property_name == "force":
            self.options.flags |= public_api.GPUXTB_COMPUTE_FORCES
        configure_gpuxtb_conformance_scc(self.options)

        systems = cell.batch_size
        atoms = molecule_atoms = cell.molecule.natoms * systems
        self.energies = (ctypes.c_double * systems)()
        self.forces = (
            (ctypes.c_double * (3 * atoms))() if cell.property_name == "force" else None
        )
        self.iterations = (ctypes.c_int32 * systems)()
        self.converged = (ctypes.c_uint8 * systems)()
        self.statuses = (ctypes.c_int32 * systems)()
        self.result = public_api.BatchResult()
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_batch_result_init(
                ctypes.byref(self.result), ctypes.sizeof(self.result)
            ),
            "gpuxtb_batch_result_init",
        )
        self.result.energies = self.memory.output(self.energies, "energies")
        if self.forces is not None:
            self.result.forces = self.memory.output(self.forces, "forces")
        self.result.scc_iterations = self.memory.output(
            self.iterations, "scc_iterations"
        )
        self.result.scc_converged = self.memory.output(self.converged, "scc_converged")
        self.result.per_system_status = self.memory.output(
            self.statuses, "per_system_status"
        )
        self.compute_options = {
            "engine": "gpuxtb",
            "model": int(self.options.model),
            "flags": int(self.options.flags),
            "max_scc_iterations": int(self.options.max_scc_iterations),
            "charge_tolerance": float(self.options.charge_tolerance),
            "energy_tolerance": float(self.options.energy_tolerance),
            "electronic_temperature_hartree": float(
                self.options.electronic_temperature
            ),
            "cpu_threads": cpu_threads,
            "device_id": device_id,
            "total_atoms": molecule_atoms,
        }
        self.closed = False

    def set_start_mode(self, mode: str) -> None:
        """Select the strict public start policy without rebuilding descriptors."""
        self.options.scc_start_mode = (
            public_api.GPUXTB_SCC_START_WARM
            if mode == "warm"
            else public_api.GPUXTB_SCC_START_FRESH
        )

    def invoke(self) -> None:
        """Execute one synchronous public batch inference."""
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_compute(
                self.context,
                ctypes.byref(self.batch),
                ctypes.byref(self.options),
                ctypes.byref(self.result),
            ),
            "natoms public batch inference",
        )

    def snapshot(self) -> dict[str, Any]:
        """Materialize one completed call after its timing boundary."""
        self.memory.download_outputs()
        return {
            "energies_hartree": [float(value) for value in self.energies],
            "scc_iterations": [int(value) for value in self.iterations],
            "scc_converged": [int(value) for value in self.converged],
            "per_system_status": [int(value) for value in self.statuses],
            "forces_hartree_per_bohr": (
                [float(value) for value in self.forces]
                if self.forces is not None
                else None
            ),
        }

    def close(self) -> None:
        """Release context and descriptor-owned device memory exactly once."""
        if self.closed:
            return
        self.closed = True
        self.memory.close()
        self.library.gpuxtb_context_destroy(self.context)


class ReferenceRunner:
    """Own one persistent tblite or xTB public-C-API adapter.

    Imports are intentionally delayed until the selected reference engine is
    constructed. A gpuxtb-only invocation therefore has no dxtb, tblite, xTB,
    PyTorch, or matplotlib import dependency.
    """

    def __init__(
        self,
        library_path: Path,
        cell: Cell,
        _cpu_threads: int,
        _device_id: int,
    ) -> None:
        storage = make_storage(cell.molecule, cell.batch_size)
        if cell.engine == "tblite":
            module_name = (
                "benchmarks.tblite_adapter" if __package__ else "tblite_adapter"
            )
            adapter_type = importlib.import_module(module_name).TbliteAdapter
            self.adapter = adapter_type(
                library_path,
                storage,
                cell.property_name,
                accuracy=1.0e-4,
                max_iterations=500,
                collect_atomic_charges=False,
            )
            self.compute_options = {
                "engine": "tblite",
                "version": self.adapter.version,
                "accuracy": self.adapter.accuracy,
                "max_scc_iterations": self.adapter.max_iterations,
                "electronic_temperature_hartree": (
                    self.adapter.electronic_temperature_hartree
                ),
                "thread_control": self.adapter.thread_control,
                "timed_result_getters": [
                    "energy",
                    *(["gradient"] if cell.property_name == "force" else []),
                ],
                "logical_batch_execution": "serial persistent public C API loop",
            }
        elif cell.engine == "xtb":
            module_name = "benchmarks.xtb_adapter" if __package__ else "xtb_adapter"
            adapter_type = importlib.import_module(module_name).XtbAdapter
            self.adapter = adapter_type(
                library_path,
                storage,
                cell.property_name,
                None,
                accuracy=1.0e-4,
                max_iterations=500,
            )
            self.compute_options = {
                "engine": "xtb",
                "api_version": self.adapter.api_version,
                "accuracy": self.adapter.accuracy,
                "max_scc_iterations": self.adapter.max_iterations,
                "electronic_temperature_kelvin": (
                    self.adapter.electronic_temperature_kelvin
                ),
                "thread_control": self.adapter.thread_control,
                "timed_result_getters": [
                    "energy",
                    *(["gradient"] if cell.property_name == "force" else []),
                ],
                "logical_batch_execution": "serial persistent public C API loop",
            }
        else:
            raise BenchmarkError(f"unsupported reference engine: {cell.engine}")
        self.mode = "persistent"
        self.closed = False

    def set_start_mode(self, mode: str) -> None:
        """References expose persistent state rather than gpuxtb start tags."""
        if mode != "persistent":
            raise BenchmarkError(
                f"reference engine cannot select gpuxtb SCC start mode {mode}"
            )
        self.mode = mode

    def invoke(self) -> None:
        """Execute one persistent reference public-C-API logical batch."""
        self.adapter.invoke()

    def snapshot(self) -> dict[str, Any]:
        """Normalize reference results after the timing boundary."""
        output = self.adapter.results()
        return {
            "energies_hartree": list(output["energies_hartree"]),
            "scc_iterations": None,
            "scc_converged": None,
            "per_system_status": None,
            "forces_hartree_per_bohr": (
                list(output["forces_hartree_per_bohr"])
                if "forces_hartree_per_bohr" in output
                else None
            ),
        }

    def close(self) -> None:
        """Release all persistent reference states exactly once."""
        if self.closed:
            return
        self.closed = True
        self.adapter.close()


def _validate_forces(
    snapshot: dict[str, Any], expected_force_count: int | None, source: str
) -> None:
    """Require the complete requested force vector, not merely a finite flag."""
    forces = snapshot.get("forces_hartree_per_bohr")
    if expected_force_count is None:
        if forces is not None:
            raise BenchmarkError(f"{source} returned unrequested forces")
        return
    if not isinstance(forces, list) or len(forces) != expected_force_count:
        actual = len(forces) if isinstance(forces, list) else None
        raise BenchmarkError(
            f"{source} returned {actual} force values; expected {expected_force_count}"
        )
    if not all(math.isfinite(value) for value in forces):
        raise BenchmarkError(f"{source} produced nonfinite requested forces")


def validate_snapshot(
    snapshot: dict[str, Any], expected_force_count: int | None
) -> None:
    """Reject data-level failures before a timing row can be called available."""
    energies = snapshot["energies_hartree"]
    if not energies or not all(math.isfinite(value) for value in energies):
        raise BenchmarkError("public inference produced nonfinite energies")
    if any(
        status != public_api.GPUXTB_STATUS_SUCCESS
        for status in snapshot["per_system_status"]
    ):
        raise BenchmarkError(
            f"public inference produced per-system failures: {snapshot['per_system_status']}"
        )
    if any(value != 1 for value in snapshot["scc_converged"]):
        raise BenchmarkError(
            f"public inference did not converge every system: {snapshot['scc_converged']}"
        )
    if any(value <= 0 for value in snapshot["scc_iterations"]):
        raise BenchmarkError(
            f"public inference reported invalid SCC iterations: {snapshot['scc_iterations']}"
        )
    _validate_forces(snapshot, expected_force_count, "public inference")


def validate_reference_snapshot(
    snapshot: dict[str, Any], expected_force_count: int | None
) -> None:
    """Reject nonfinite reference output without inventing SCC diagnostics."""
    energies = snapshot["energies_hartree"]
    if not energies or not all(math.isfinite(value) for value in energies):
        raise BenchmarkError("reference public inference produced nonfinite energies")
    _validate_forces(snapshot, expected_force_count, "reference public inference")


def timing_summary(samples_ms: Sequence[float], batch_size: int) -> dict[str, Any]:
    """Summarize without discarding the raw latency evidence."""
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    median = statistics.median(samples_ms)
    return {
        "samples_ms": list(samples_ms),
        "count": len(samples_ms),
        "min_ms": min(samples_ms),
        "median_ms": median,
        "mean_ms": statistics.fmean(samples_ms),
        "p95_ms": ordered[p95_index],
        "systems_per_second_at_median": 1000.0 * batch_size / median,
    }


def measure_runner(
    runner: Runner,
    protocol: Protocol,
    batch_size: int,
    expected_force_count: int | None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run the strict seed/warmup/sample sequence on one persistent runner."""
    seed_snapshot = None
    if protocol.start_mode == "warm":
        runner.set_start_mode("fresh")
        runner.invoke()
        seed_snapshot = runner.snapshot()
        validate_snapshot(seed_snapshot, expected_force_count)
    runner.set_start_mode(protocol.start_mode)
    for _ in range(protocol.warmups):
        runner.invoke()
        validate_snapshot(runner.snapshot(), expected_force_count)

    raw_samples = []
    for sample_index in range(protocol.repetitions):
        start = clock_ns()
        runner.invoke()
        elapsed_ms = (clock_ns() - start) * 1.0e-6
        snapshot = runner.snapshot()
        validate_snapshot(snapshot, expected_force_count)
        raw_samples.append(
            {
                "sample_index": sample_index,
                "start_mode": protocol.start_mode,
                "latency_ms": elapsed_ms,
                **snapshot,
            }
        )

    reference_energies = (
        seed_snapshot["energies_hartree"]
        if seed_snapshot is not None
        else raw_samples[0]["energies_hartree"]
    )
    energy_drift = max(
        abs(actual - expected)
        for sample in raw_samples
        for actual, expected in zip(sample["energies_hartree"], reference_energies)
    )
    reference_forces = (
        seed_snapshot["forces_hartree_per_bohr"]
        if seed_snapshot is not None
        else raw_samples[0]["forces_hartree_per_bohr"]
    )
    force_drift = None
    if expected_force_count is not None:
        force_drift = max(
            abs(actual - expected)
            for sample in raw_samples
            for actual, expected in zip(
                sample["forces_hartree_per_bohr"], reference_forces
            )
        )
    iteration_nonregression = True
    if seed_snapshot is not None:
        iteration_nonregression = all(
            warm <= fresh
            for sample in raw_samples
            for warm, fresh in zip(
                sample["scc_iterations"], seed_snapshot["scc_iterations"]
            )
        )
    correctness_passed = (
        energy_drift <= protocol.energy_atol_hartree
        and (force_drift is None or force_drift <= protocol.force_atol_hartree_per_bohr)
        and iteration_nonregression
    )
    latencies = [sample["latency_ms"] for sample in raw_samples]
    all_energies = [
        energy for sample in raw_samples for energy in sample["energies_hartree"]
    ]
    all_iterations = [
        iteration for sample in raw_samples for iteration in sample["scc_iterations"]
    ]
    return {
        "seed": seed_snapshot,
        "raw_samples": raw_samples,
        "timing": timing_summary(latencies, batch_size),
        "energy_summary": {
            "min_hartree": min(all_energies),
            "max_hartree": max(all_energies),
        },
        "iteration_summary": {
            "min": min(all_iterations),
            "median": statistics.median(all_iterations),
            "max": max(all_iterations),
        },
        "correctness": {
            "status": "pass" if correctness_passed else "fail",
            "energy_reference_hartree": reference_energies,
            "energy_atol_hartree": protocol.energy_atol_hartree,
            "energy_tolerance_source": (
                PRIMARY_ENERGY_TOLERANCE_SOURCE
                if protocol.energy_atol_hartree == DEFAULT_ENERGY_ATOL_LIMIT
                else {"kind": "explicit_stricter_gate"}
            ),
            "max_abs_energy_drift_hartree": energy_drift,
            "force_reference_hartree_per_bohr": reference_forces,
            "force_atol_hartree_per_bohr": protocol.force_atol_hartree_per_bohr,
            "max_abs_force_drift_hartree_per_bohr": force_drift,
            "warm_iterations_no_greater_than_fresh_seed": iteration_nonregression,
        },
    }


def measure_reference_runner(
    runner: Runner,
    protocol: Protocol,
    batch_size: int,
    expected_force_count: int | None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Record one first-call cold sample and persistent steady-state samples."""
    runner.set_start_mode("persistent")
    cold_start = clock_ns()
    runner.invoke()
    cold_latency_ms = (clock_ns() - cold_start) * 1.0e-6
    cold_snapshot = runner.snapshot()
    validate_reference_snapshot(cold_snapshot, expected_force_count)
    for _ in range(protocol.warmups):
        runner.invoke()
        validate_reference_snapshot(runner.snapshot(), expected_force_count)

    raw_samples = []
    for sample_index in range(protocol.repetitions):
        start = clock_ns()
        runner.invoke()
        elapsed_ms = (clock_ns() - start) * 1.0e-6
        snapshot = runner.snapshot()
        validate_reference_snapshot(snapshot, expected_force_count)
        raw_samples.append(
            {
                "sample_index": sample_index,
                "start_mode": "persistent",
                "latency_ms": elapsed_ms,
                **snapshot,
            }
        )

    reference_energies = cold_snapshot["energies_hartree"]
    energy_drift = max(
        abs(actual - expected)
        for sample in raw_samples
        for actual, expected in zip(sample["energies_hartree"], reference_energies)
    )
    reference_forces = cold_snapshot["forces_hartree_per_bohr"]
    force_drift = None
    if expected_force_count is not None:
        force_drift = max(
            abs(actual - expected)
            for sample in raw_samples
            for actual, expected in zip(
                sample["forces_hartree_per_bohr"], reference_forces
            )
        )
    correctness_passed = energy_drift <= protocol.energy_atol_hartree and (
        force_drift is None or force_drift <= protocol.force_atol_hartree_per_bohr
    )
    latencies = [sample["latency_ms"] for sample in raw_samples]
    all_energies = [
        energy for sample in raw_samples for energy in sample["energies_hartree"]
    ]
    return {
        "cold_sample": {
            "latency_ms": cold_latency_ms,
            **cold_snapshot,
        },
        "seed": None,
        "raw_samples": raw_samples,
        "timing": timing_summary(latencies, batch_size),
        "energy_summary": {
            "min_hartree": min(all_energies),
            "max_hartree": max(all_energies),
        },
        "iteration_summary": None,
        "correctness": {
            "status": "pass" if correctness_passed else "fail",
            "energy_reference_hartree": reference_energies,
            "energy_atol_hartree": protocol.energy_atol_hartree,
            "max_abs_energy_drift_hartree": energy_drift,
            "force_reference_hartree_per_bohr": reference_forces,
            "force_atol_hartree_per_bohr": protocol.force_atol_hartree_per_bohr,
            "max_abs_force_drift_hartree_per_bohr": force_drift,
            "warm_iterations_no_greater_than_fresh_seed": None,
        },
    }


def execute_cell(
    cell: Cell,
    protocol: Protocol,
    library: Path,
    cpu_threads: int,
    device_id: int,
    run_identity: dict[str, Any],
    runner_factory: Callable[[Path, Cell, int, int], Runner] | None = None,
) -> dict[str, Any]:
    """Execute one cell and attach all row-level provenance."""
    factory = runner_factory or (
        GpuxtbRunner if cell.engine == "gpuxtb" else ReferenceRunner
    )
    runner = factory(library, cell, cpu_threads, device_id)
    try:
        expected_force_count = (
            3 * cell.molecule.natoms * cell.batch_size
            if cell.property_name == "force"
            else None
        )
        measured = (
            measure_runner(runner, protocol, cell.batch_size, expected_force_count)
            if cell.engine == "gpuxtb"
            else measure_reference_runner(
                runner, protocol, cell.batch_size, expected_force_count
            )
        )
        return {
            "availability": "available",
            "engine": cell.engine,
            "backend": cell.backend,
            "memory_mode": "host",
            "molecule": cell.molecule.name,
            "natoms": cell.molecule.natoms,
            "batch_size": cell.batch_size,
            "property": cell.property_name,
            "start_mode": protocol.start_mode,
            "warmups": protocol.warmups,
            "repetitions": protocol.repetitions,
            "compute_options": runner.compute_options,
            "workload_identity": workload_identity(cell),
            "run_identity": run_identity,
            **measured,
        }
    finally:
        runner.close()


def collect_rows(
    cells: Sequence[Cell],
    protocol: Protocol,
    library: Path,
    cpu_threads: int,
    device_id: int,
    run_identity: dict[str, Any],
    runner_factory: Callable[[Path, Cell, int, int], Runner] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Retain explicit error rows while allowing independent cells to finish."""
    rows = []
    failed = False
    for cell in cells:
        try:
            row = execute_cell(
                cell,
                protocol,
                library,
                cpu_threads,
                device_id,
                run_identity,
                runner_factory,
            )
        except Exception as exc:  # noqa: BLE001 - artifact preserves the failure
            failed = True
            row = {
                "availability": "error",
                "error": str(exc),
                "engine": cell.engine,
                "backend": cell.backend,
                "memory_mode": "host",
                "molecule": cell.molecule.name,
                "natoms": cell.molecule.natoms,
                "batch_size": cell.batch_size,
                "property": cell.property_name,
                "start_mode": protocol.start_mode,
                "warmups": protocol.warmups,
                "repetitions": protocol.repetitions,
                "workload_identity": workload_identity(cell),
                "run_identity": run_identity,
            }
        if row.get("correctness", {}).get("status") == "fail":
            failed = True
        rows.append(row)
    return rows, failed


def _validated_reference_options(
    options: Any, property_name: str, natoms: int, batch_size: int
) -> dict[str, Any]:
    """Validate and retain the complete gpuxtb option identity from one row."""
    required = {
        "engine",
        "model",
        "flags",
        "max_scc_iterations",
        "charge_tolerance",
        "energy_tolerance",
        "electronic_temperature_hartree",
        "cpu_threads",
        "device_id",
        "total_atoms",
    }
    if not isinstance(options, dict) or set(options) != required:
        raise BenchmarkError(
            "reference row compute_options must contain the complete gpuxtb option image"
        )
    expected_flags = public_api.GPUXTB_COMPUTE_ENERGY
    if property_name == "force":
        expected_flags |= public_api.GPUXTB_COMPUTE_FORCES
    if (
        options["engine"] != "gpuxtb"
        or options["model"] != public_api.GPUXTB_MODEL_GFN2_XTB
        or options["flags"] != expected_flags
        or options["total_atoms"] != natoms * batch_size
        or type(options["cpu_threads"]) is not int
        or options["cpu_threads"] <= 0
        or type(options["device_id"]) is not int
        or options["device_id"] < 0
    ):
        raise BenchmarkError("reference row has inconsistent gpuxtb compute options")
    numeric_fields = (
        "charge_tolerance",
        "energy_tolerance",
        "electronic_temperature_hartree",
    )
    if not all(
        type(options[name]) in (int, float) and math.isfinite(options[name])
        for name in numeric_fields
    ):
        raise BenchmarkError("reference row has nonfinite gpuxtb compute options")
    if type(options["max_scc_iterations"]) is not int:
        raise BenchmarkError("reference row max_scc_iterations is not an integer")
    if (
        options["max_scc_iterations"] != GPUXTB_CONFORMANCE_MAX_SCC_ITERATIONS
        or options["charge_tolerance"] != GPUXTB_CONFORMANCE_CHARGE_TOLERANCE
        or options["energy_tolerance"] != GPUXTB_CONFORMANCE_ENERGY_TOLERANCE
        or options["electronic_temperature_hartree"]
        != GPUXTB_CONFORMANCE_ELECTRONIC_TEMPERATURE
    ):
        raise BenchmarkError(
            "reference row does not use the pinned gpuxtb conformance SCC options"
        )
    return dict(options)


def _validated_finite_vector(
    value: Any,
    expected_count: int,
    field_name: str,
    source: str = "reference row",
) -> tuple[float, ...]:
    """Normalize one exact-length finite reference vector."""
    if not isinstance(value, list) or len(value) != expected_count:
        raise BenchmarkError(
            f"{source} {field_name} must contain {expected_count} values"
        )
    if not all(type(item) in (int, float) and math.isfinite(item) for item in value):
        raise BenchmarkError(
            f"{source} {field_name} contains nonnumeric or nonfinite values"
        )
    return tuple(float(item) for item in value)


def _validated_raw_sample_vectors(
    value: Any,
    expected_repetitions: int,
    expected_energy_count: int,
    expected_force_count: int | None,
    source: str,
    expected_start_mode: str | None = None,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...] | None]:
    """Validate and normalize every measured observable vector in one row."""
    if not isinstance(value, list) or len(value) != expected_repetitions:
        raise BenchmarkError(
            f"{source} raw_samples must contain {expected_repetitions} measured samples"
        )
    energy_samples: list[tuple[float, ...]] = []
    force_samples: list[tuple[float, ...]] = []
    for sample_index, sample in enumerate(value):
        sample_source = f"{source} raw sample {sample_index}"
        if not isinstance(sample, dict):
            raise BenchmarkError(f"{sample_source} is not an object")
        if (
            type(sample.get("sample_index")) is not int
            or sample["sample_index"] != sample_index
        ):
            raise BenchmarkError(f"{sample_source} has a nonconsecutive sample_index")
        if (
            expected_start_mode is not None
            and sample.get("start_mode") != expected_start_mode
        ):
            raise BenchmarkError(
                f"{sample_source} does not use {expected_start_mode.upper()} start mode"
            )
        energy_samples.append(
            _validated_finite_vector(
                sample.get("energies_hartree"),
                expected_energy_count,
                "energies_hartree",
                sample_source,
            )
        )
        if expected_force_count is None:
            if sample.get("forces_hartree_per_bohr") is not None:
                raise BenchmarkError(f"{sample_source} contains unrequested forces")
        else:
            force_samples.append(
                _validated_finite_vector(
                    sample.get("forces_hartree_per_bohr"),
                    expected_force_count,
                    "forces_hartree_per_bohr",
                    sample_source,
                )
            )
    return tuple(energy_samples), (tuple(force_samples) if force_samples else None)


def load_reference_artifact(path: Path) -> ReferenceArtifact:
    """Read, hash, parse, and strictly validate one FRESH artifact exactly once."""
    resolved = path.resolve()
    try:
        artifact_bytes = resolved.read_bytes()
        document = json.loads(artifact_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"invalid reference artifact {path}: {exc}") from exc
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    if not isinstance(document, dict):
        raise BenchmarkError("reference artifact top level must be an object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkError(f"reference artifact schema must be {SCHEMA_VERSION}")
    if document.get("engine") != "gpuxtb":
        raise BenchmarkError("reference artifact engine must be gpuxtb")
    protocol = document.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("start_mode") != "fresh":
        raise BenchmarkError("reference artifact protocol must use FRESH start mode")
    protocol_repetitions = protocol.get("repetitions")
    if type(protocol_repetitions) is not int or protocol_repetitions <= 0:
        raise BenchmarkError("reference artifact protocol repetitions must be positive")
    protocol_energy_atol = protocol.get("energy_atol_hartree")
    protocol_force_atol = protocol.get("force_atol_hartree_per_bohr")
    if (
        type(protocol_energy_atol) not in (int, float)
        or not math.isfinite(protocol_energy_atol)
        or protocol_energy_atol < 0.0
        or protocol_energy_atol > DEFAULT_ENERGY_ATOL_LIMIT
        or type(protocol_force_atol) not in (int, float)
        or not math.isfinite(protocol_force_atol)
        or protocol_force_atol < 0.0
        or protocol_force_atol > DEFAULT_FORCE_ATOL
    ):
        raise BenchmarkError(
            "reference artifact protocol uses correctness gates wider than the manifest"
        )
    run_identity = document.get("run_identity")
    rows = document.get("rows")
    if not isinstance(run_identity, dict) or not isinstance(rows, list) or not rows:
        raise BenchmarkError("reference artifact requires run_identity and rows")
    _gpuxtb_binary_source_identity(run_identity, "reference artifact")

    references: dict[str, ReferenceRow] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BenchmarkError(f"reference row {index} is not an object")
        correctness = row.get("correctness")
        if not isinstance(correctness, dict):
            raise BenchmarkError(f"reference row {index} correctness is not an object")
        if (
            row.get("engine") != "gpuxtb"
            or row.get("start_mode") != "fresh"
            or row.get("repetitions") != protocol_repetitions
            or row.get("availability") != "available"
            or row.get("run_identity") != run_identity
            or correctness.get("status") != "pass"
            or correctness.get("energy_atol_hartree") != protocol_energy_atol
            or correctness.get("force_atol_hartree_per_bohr") != protocol_force_atol
        ):
            raise BenchmarkError(
                f"reference row {index} is not a successful gpuxtb FRESH row"
            )
        try:
            natoms = int(row["natoms"])
            batch_size = int(row["batch_size"])
            backend = str(row["backend"])
            property_name = str(row["property"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkError(
                f"reference row {index} has invalid workload fields"
            ) from exc
        if (
            batch_size <= 0
            or backend not in ("cpu", "cuda")
            or property_name not in ("energy", "force")
        ):
            raise BenchmarkError(f"reference row {index} has unsupported workload tags")
        expected_identity = workload_identity(
            Cell(
                "gpuxtb",
                make_alkane(natoms),
                batch_size,
                backend,
                property_name,
            )
        )
        if row.get("workload_identity") != expected_identity:
            raise BenchmarkError(
                f"reference row {index} workload identity is incomplete or inconsistent"
            )
        if any(
            row.get(name) != expected_identity[name]
            for name in (
                "molecule",
                "natoms",
                "batch_size",
                "backend",
                "memory_mode",
                "property",
            )
        ):
            raise BenchmarkError(
                f"reference row {index} duplicates inconsistent workload fields"
            )
        options = _validated_reference_options(
            row.get("compute_options"), property_name, natoms, batch_size
        )
        energy_samples, force_samples = _validated_raw_sample_vectors(
            row.get("raw_samples"),
            protocol_repetitions,
            batch_size,
            3 * natoms * batch_size if property_name == "force" else None,
            f"reference row {index}",
            expected_start_mode="fresh",
        )
        energy_drift = correctness.get("max_abs_energy_drift_hartree")
        force_drift = correctness.get("max_abs_force_drift_hartree_per_bohr")
        if (
            type(energy_drift) not in (int, float)
            or not math.isfinite(energy_drift)
            or energy_drift < 0.0
            or energy_drift > protocol_energy_atol
            or (
                property_name == "force"
                and (
                    type(force_drift) not in (int, float)
                    or not math.isfinite(force_drift)
                    or force_drift < 0.0
                    or force_drift > protocol_force_atol
                )
            )
            or (property_name == "energy" and force_drift is not None)
        ):
            raise BenchmarkError(
                f"reference row {index} drift does not satisfy its strict protocol"
            )
        energies = _validated_finite_vector(
            correctness.get("energy_reference_hartree"),
            batch_size,
            "energy_reference_hartree",
        )
        calculated_energy_drift = max(
            abs(observed - reference)
            for sample in energy_samples
            for observed, reference in zip(sample, energy_samples[0])
        )
        if energies != energy_samples[0] or energy_drift != calculated_energy_drift:
            raise BenchmarkError(
                f"reference row {index} energy summary does not match raw samples"
            )
        forces = None
        if property_name == "force":
            forces = _validated_finite_vector(
                correctness.get("force_reference_hartree_per_bohr"),
                3 * natoms * batch_size,
                "force_reference_hartree_per_bohr",
            )
            assert force_samples is not None
            calculated_force_drift = max(
                abs(observed - reference)
                for sample in force_samples
                for observed, reference in zip(sample, force_samples[0])
            )
            if forces != force_samples[0] or force_drift != calculated_force_drift:
                raise BenchmarkError(
                    f"reference row {index} force summary does not match raw samples"
                )
        elif correctness.get("force_reference_hartree_per_bohr") is not None:
            raise BenchmarkError(
                f"reference row {index} contains forces for an energy-only workload"
            )
        key = workload_key(expected_identity)
        if key in references:
            raise BenchmarkError(f"reference artifact contains duplicate row key {key}")
        references[key] = ReferenceRow(energies, forces, options)
    return ReferenceArtifact(resolved, digest, run_identity, references)


def apply_cross_engine_correctness(
    rows: Sequence[dict[str, Any]],
    reference_artifact: ReferenceArtifact | None,
    energy_atol_hartree: float,
    force_atol_hartree_per_bohr: float,
) -> bool:
    """Compare complete observable results against one validated FRESH artifact."""
    failed = False
    for row in rows:
        if row.get("availability") != "available":
            continue
        comparison = {
            "status": "not_requested",
            "artifact": (str(reference_artifact.path) if reference_artifact else None),
            "artifact_sha256": (
                reference_artifact.sha256 if reference_artifact else None
            ),
            "tolerance_source": (
                CROSS_ENGINE_TOLERANCE_SOURCE
                if energy_atol_hartree == DEFAULT_CROSS_ENGINE_ENERGY_ATOL
                and force_atol_hartree_per_bohr == DEFAULT_CROSS_ENGINE_FORCE_ATOL
                else {
                    "kind": "explicit_cli_override",
                    "committed_defaults": CROSS_ENGINE_TOLERANCE_SOURCE,
                }
            ),
            "energy": {
                "atol_hartree": energy_atol_hartree,
                "max_abs_delta_hartree": None,
            },
            "force": {
                "atol_hartree_per_bohr": force_atol_hartree_per_bohr,
                "max_abs_delta_hartree_per_bohr": None,
                "status": "not_requested",
            },
            "measured_samples": {"status": "not_requested", "count": None},
            "gpuxtb_option_identity": "not_comparable_cross_engine",
            "gpuxtb_binary_identity": "not_comparable_cross_engine",
            "gpuxtb_repository_revision": "not_comparable_cross_engine",
        }
        if reference_artifact is not None:
            expected = reference_artifact.rows.get(
                workload_key(row["workload_identity"])
            )
            if expected is None:
                comparison["status"] = "missing_reference"
                failed = True
            else:
                expected_force_count = (
                    len(expected.forces_hartree_per_bohr)
                    if row["property"] == "force"
                    and expected.forces_hartree_per_bohr is not None
                    else None
                )
                repetitions = row.get("repetitions")
                try:
                    if type(repetitions) is not int or repetitions <= 0:
                        raise BenchmarkError(
                            "dependent row repetitions must be a positive integer"
                        )
                    energy_samples, force_samples = _validated_raw_sample_vectors(
                        row.get("raw_samples"),
                        repetitions,
                        len(expected.energies_hartree),
                        expected_force_count,
                        "dependent row",
                    )
                except BenchmarkError as exc:
                    energy_passed = False
                    force_passed = row["property"] != "force"
                    comparison["measured_samples"] = {
                        "status": "fail",
                        "count": (
                            len(row["raw_samples"])
                            if isinstance(row.get("raw_samples"), list)
                            else None
                        ),
                        "error": str(exc),
                    }
                    if row["property"] == "force":
                        comparison["force"]["status"] = "fail"
                else:
                    energy_delta = max(
                        abs(observed - reference)
                        for sample in energy_samples
                        for observed, reference in zip(
                            sample, expected.energies_hartree
                        )
                    )
                    comparison["energy"]["max_abs_delta_hartree"] = energy_delta
                    energy_passed = energy_delta <= energy_atol_hartree
                    force_passed = True
                    if row["property"] == "force":
                        assert expected.forces_hartree_per_bohr is not None
                        assert force_samples is not None
                        force_delta = max(
                            abs(observed - reference)
                            for sample in force_samples
                            for observed, reference in zip(
                                sample, expected.forces_hartree_per_bohr
                            )
                        )
                        comparison["force"]["max_abs_delta_hartree_per_bohr"] = (
                            force_delta
                        )
                        force_passed = force_delta <= force_atol_hartree_per_bohr
                        comparison["force"]["status"] = (
                            "pass" if force_passed else "fail"
                        )
                    comparison["measured_samples"] = {
                        "status": "pass",
                        "count": repetitions,
                    }
                option_passed = True
                binary_passed = True
                revision_passed = True
                if row["engine"] == "gpuxtb":
                    option_passed = row["compute_options"] == expected.compute_options
                    comparison["gpuxtb_option_identity"] = (
                        "pass" if option_passed else "fail"
                    )
                    current_identity = row["run_identity"]
                    try:
                        current_sha, current_revision = _gpuxtb_binary_source_identity(
                            current_identity, "dependent gpuxtb row"
                        )
                    except BenchmarkError:
                        binary_passed = False
                        revision_passed = False
                    else:
                        reference_sha = reference_artifact.run_identity["library"][
                            "sha256"
                        ]
                        reference_revision = reference_artifact.run_identity[
                            "repository"
                        ]["revision"]
                        binary_passed = current_sha == reference_sha
                        revision_passed = current_revision == reference_revision
                    comparison["gpuxtb_binary_identity"] = (
                        "pass" if binary_passed else "fail"
                    )
                    comparison["gpuxtb_repository_revision"] = (
                        "pass" if revision_passed else "fail"
                    )
                comparison["status"] = (
                    "pass"
                    if energy_passed
                    and force_passed
                    and option_passed
                    and binary_passed
                    and revision_passed
                    else "fail"
                )
                failed = failed or comparison["status"] == "fail"
        row["correctness"]["fresh_reference_comparison"] = comparison
    return failed


def build_document(
    rows: Sequence[dict[str, Any]],
    run_identity: dict[str, Any],
    protocol: Protocol,
) -> dict[str, Any]:
    """Build one self-contained JSON artifact without external row discovery."""
    engine = rows[0]["engine"] if rows else None
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "run_identity": run_identity,
        "protocol": {
            "start_mode": protocol.start_mode,
            "warmups": protocol.warmups,
            "repetitions": protocol.repetitions,
            "energy_atol_hartree": protocol.energy_atol_hartree,
            "force_atol_hartree_per_bohr": protocol.force_atol_hartree_per_bohr,
            "force_tolerance_source": (
                PRIMARY_FORCE_TOLERANCE_SOURCE
                if protocol.force_atol_hartree_per_bohr == DEFAULT_FORCE_ATOL
                else {"kind": "explicit_stricter_gate"}
            ),
            "timing_scope": (
                "one synchronous gpuxtb_compute public-C-ABI call; persistent context, "
                "descriptors, options, and caller-owned result buffers; result inspection "
                "is outside the measured interval"
                if engine == "gpuxtb"
                else "one persistent reference public-C-API logical batch; setup and "
                "result inspection are outside the measured interval"
            ),
            "warm_seed": (
                "exactly one untimed FRESH call on the same identity before WARM warmups"
                if protocol.start_mode == "warm"
                else None
            ),
            "reference_cold_sample": (
                "the first persistent public-C-API invocation is recorded separately"
                if engine in ("tblite", "xtb")
                else None
            ),
        },
        "rows": list(rows),
    }


def validate_output_paths(
    json_path: Path, csv_path: Path, allow_overwrite: bool
) -> None:
    """Fail before measurement when outputs collide or would overwrite evidence."""
    if json_path.resolve() == csv_path.resolve():
        raise BenchmarkError("JSON and CSV output paths must be distinct")
    existence = (json_path.exists(), csv_path.exists())
    if allow_overwrite and any(existence) and not all(existence):
        raise FileExistsError(
            "refusing to replace an incomplete/stale benchmark artifact pair"
        )
    if not allow_overwrite:
        existing = [str(path) for path in (json_path, csv_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing benchmark artifacts: "
                + ", ".join(existing)
            )


def _serialize_csv(document: dict[str, Any]) -> str:
    """Build the complete CSV image before any output path is published."""
    rows = document["rows"]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
        )
    return output.getvalue()


def _write_unique_temporary(path: Path, content: str) -> Path:
    """Write and fsync one unique same-directory staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _pair_lock_path(json_path: Path, csv_path: Path) -> Path:
    """Name one exclusive reservation shared by every writer of this pair."""
    identity = "\0".join((str(json_path.resolve()), str(csv_path.resolve())))
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return json_path.parent / f".{json_path.name}.{suffix}.pair.lock"


def _remove_hard_link(path: Path, staged: Path) -> None:
    """Roll back only a final path that still names our staged inode."""
    try:
        final_stat = path.stat()
        staged_stat = staged.stat()
    except FileNotFoundError:
        return
    if (final_stat.st_dev, final_stat.st_ino) == (
        staged_stat.st_dev,
        staged_stat.st_ino,
    ):
        path.unlink()


def _publish_new_pair(
    json_path: Path, csv_path: Path, json_stage: Path, csv_stage: Path
) -> None:
    """Publish a new pair with exclusive hard links and complete rollback."""
    published: list[tuple[Path, Path]] = []
    try:
        for final_path, staged_path in (
            (json_path, json_stage),
            (csv_path, csv_stage),
        ):
            os.link(staged_path, final_path)
            published.append((final_path, staged_path))
    except BaseException:
        for final_path, staged_path in reversed(published):
            _remove_hard_link(final_path, staged_path)
        raise


def _unused_backup_path(path: Path) -> Path:
    """Reserve a unique same-directory backup name without leaving a file."""
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".backup"
    )
    os.close(descriptor)
    backup = Path(name)
    backup.unlink()
    return backup


def _replace_pair(
    json_path: Path, csv_path: Path, json_stage: Path, csv_stage: Path
) -> None:
    """Replace an existing pair and restore both originals after any failure."""
    finals = (json_path, csv_path)
    stages = (json_stage, csv_stage)
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for final_path in finals:
            backup = _unused_backup_path(final_path)
            os.replace(final_path, backup)
            backups[final_path] = backup
        for final_path, staged_path in zip(finals, stages):
            os.replace(staged_path, final_path)
            published.append(final_path)
    except BaseException:
        for final_path in published:
            final_path.unlink(missing_ok=True)
        for final_path, backup in backups.items():
            if backup.exists():
                os.replace(backup, final_path)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def write_artifacts(
    json_path: Path,
    csv_path: Path,
    document: dict[str, Any],
    allow_overwrite: bool,
) -> None:
    """Publish a JSON/CSV pair with exclusive reservation and failure rollback."""
    json_content = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    csv_content = _serialize_csv(document)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _pair_lock_path(json_path, csv_path)
    try:
        lock_descriptor = os.open(
            lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"benchmark artifact pair is reserved by a concurrent or stale writer: {lock_path}"
        ) from exc
    try:
        lock_stat = os.fstat(lock_descriptor)
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
    except BaseException:
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)
        raise
    else:
        os.close(lock_descriptor)
    json_stage: Path | None = None
    csv_stage: Path | None = None
    try:
        validate_output_paths(json_path, csv_path, allow_overwrite)
        json_stage = _write_unique_temporary(json_path, json_content)
        csv_stage = _write_unique_temporary(csv_path, csv_content)
        if allow_overwrite:
            _replace_pair(json_path, csv_path, json_stage, csv_stage)
        else:
            _publish_new_pair(json_path, csv_path, json_stage, csv_stage)
    finally:
        if json_stage is not None:
            json_stage.unlink(missing_ok=True)
        if csv_stage is not None:
            csv_stage.unlink(missing_ok=True)
        try:
            current_lock = lock_path.stat()
        except FileNotFoundError:
            pass
        else:
            if (current_lock.st_dev, current_lock.st_ino) == (
                lock_stat.st_dev,
                lock_stat.st_ino,
            ):
                lock_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    """Create the strict gpuxtb-only natoms CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--engine", choices=("gpuxtb", "tblite", "xtb"), default="gpuxtb"
    )
    parser.add_argument("--start-mode", choices=("fresh", "warm"))
    parser.add_argument("--natoms", type=parse_csv_ints, default=DEFAULT_NATOMS)
    parser.add_argument("--batch-sizes", type=parse_csv_ints, default=(1,))
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--property", choices=("energy", "force"), default="force")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--energy-atol", type=float, default=1.0e-8)
    parser.add_argument("--force-atol", type=float, default=DEFAULT_FORCE_ATOL)
    parser.add_argument("--energy-reference-json", type=Path)
    parser.add_argument(
        "--cross-engine-energy-atol",
        type=float,
        default=DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
    )
    parser.add_argument(
        "--cross-engine-force-atol",
        type=float,
        default=DEFAULT_CROSS_ENGINE_FORCE_ATOL,
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--allow-dirty-evidence",
        action="store_true",
        help="development only: run from dirty sources and mark artifacts ineligible",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid protocol values before opening the library."""
    if not args.library.is_file():
        raise BenchmarkError(f"selected engine library is missing: {args.library}")
    if args.engine == "gpuxtb" and args.start_mode is None:
        raise BenchmarkError("gpuxtb runs require explicit --start-mode fresh or warm")
    if args.engine != "gpuxtb" and args.start_mode is not None:
        raise BenchmarkError("--start-mode applies only to the gpuxtb engine")
    if args.engine != "gpuxtb" and args.backend != "cpu":
        raise BenchmarkError(
            "tblite and xTB natoms references support only --backend cpu"
        )
    if (
        args.engine != "gpuxtb" or args.start_mode == "warm"
    ) and args.energy_reference_json is None:
        raise BenchmarkError(
            "gpuxtb WARM and reference-engine runs require --energy-reference-json"
        )
    if args.cpu_threads <= 0:
        raise BenchmarkError("--cpu-threads must be positive")
    if args.device_id < 0:
        raise BenchmarkError("--device-id must be nonnegative")
    if args.warmups < 0:
        raise BenchmarkError("--warmups must be nonnegative")
    if args.repetitions <= 0:
        raise BenchmarkError("--repetitions must be positive")
    if (
        not math.isfinite(args.energy_atol)
        or args.energy_atol < 0.0
        or args.energy_atol > DEFAULT_ENERGY_ATOL_LIMIT
    ):
        raise BenchmarkError(
            f"--energy-atol must be between 0 and {DEFAULT_ENERGY_ATOL_LIMIT:g}"
        )
    if (
        not math.isfinite(args.force_atol)
        or args.force_atol < 0.0
        or args.force_atol > DEFAULT_FORCE_ATOL
    ):
        raise BenchmarkError(
            f"--force-atol must be between 0 and {DEFAULT_FORCE_ATOL:g}"
        )
    if (
        not math.isfinite(args.cross_engine_energy_atol)
        or args.cross_engine_energy_atol < 0.0
        or args.cross_engine_energy_atol > DEFAULT_CROSS_ENGINE_ENERGY_ATOL
    ):
        raise BenchmarkError(
            "--cross-engine-energy-atol cannot exceed the committed manifest gate"
        )
    if (
        not math.isfinite(args.cross_engine_force_atol)
        or args.cross_engine_force_atol < 0.0
        or args.cross_engine_force_atol > DEFAULT_CROSS_ENGINE_FORCE_ATOL
    ):
        raise BenchmarkError(
            "--cross-engine-force-atol cannot exceed the committed manifest gate"
        )
    if (
        args.energy_reference_json is not None
        and not args.energy_reference_json.is_file()
    ):
        raise BenchmarkError(
            f"FRESH reference artifact is missing: {args.energy_reference_json}"
        )
    for natoms in args.natoms:
        make_alkane(natoms)
    validate_output_paths(args.output_json, args.output_csv, args.allow_overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    """Run all requested cells and leave a complete artifact even on cell errors."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    try:
        validate_arguments(args)
        library = args.library.resolve()
        reference_artifact = (
            load_reference_artifact(args.energy_reference_json)
            if args.energy_reference_json is not None
            else None
        )
        identity = collect_run_identity(
            args.engine, library, arguments, reference_artifact
        )
        apply_current_evidence_policy(identity, args.allow_dirty_evidence)
        protocol = Protocol(
            args.start_mode if args.engine == "gpuxtb" else "persistent",
            args.warmups,
            args.repetitions,
            args.energy_atol,
            args.force_atol,
        )
        molecules = [make_alkane(natoms) for natoms in args.natoms]
        cells = [
            Cell(args.engine, molecule, batch_size, args.backend, args.property)
            for molecule in molecules
            for batch_size in args.batch_sizes
        ]
        rows, failed = collect_rows(
            cells,
            protocol,
            library,
            args.cpu_threads,
            args.device_id,
            identity,
        )
        failed = identity["evidence_eligibility"]["status"] != "eligible" or failed
        failed = (
            apply_cross_engine_correctness(
                rows,
                reference_artifact,
                args.cross_engine_energy_atol,
                args.cross_engine_force_atol,
            )
            or failed
        )
        document = build_document(rows, identity, protocol)
        write_artifacts(
            args.output_json,
            args.output_csv,
            document,
            args.allow_overwrite,
        )
    except (BenchmarkError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for row in rows:
        if row["availability"] == "available":
            print(
                f"{row['molecule']} natoms={row['natoms']} batch={row['batch_size']} "
                f"{row['start_mode']} median={row['timing']['median_ms']:.6f} ms "
                f"iterations={row['raw_samples'][-1]['scc_iterations']}"
            )
        else:
            print(
                f"ERROR {row['molecule']} batch={row['batch_size']}: {row['error']}",
                file=sys.stderr,
            )
    print(f"wrote {args.output_json} and {args.output_csv}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
